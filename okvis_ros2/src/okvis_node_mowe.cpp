/**
 * @file okvis_node_mowe.cpp
 * @brief Mow-e bench VIO node: owns the OV9281 stereo pair directly via
 *        mowe_camera_core and feeds OKVIS through ViInterface::addImages.
 *
 * The image hot path has NO ROS in it (ADR-0040 §2): the camera delivers both
 * eyes as ONE hardware-synchronized FrameBundle (single V4L2 frame, single
 * CLOCK_MONOTONIC SOF stamp), so the stereo pair can never split, drop one
 * half, or mis-pair — the failure modes of the two-topic okvis_node_subscriber
 * path. It also sidesteps the mowe_camera lifecycle-activation trap entirely.
 *
 * IMU still arrives over ROS from the sch16t node (small messages; fine over
 * DDS), and pose/visualisation go out via okvis::Publisher. Because the SCH16T
 * node is a LifecycleNode whose launch-side auto configure/activate is
 * unreliable, this node drives the transitions itself at startup (parameter
 * `activate_lifecycle_node`).
 */
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include <okvis/ThreadedSlam.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ros2/Publisher.hpp>

#include <lifecycle_msgs/msg/transition.hpp>
#include <lifecycle_msgs/srv/change_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "mowe_camera/camera.hpp"
#include "mowe_camera/factory.hpp"
#include "mowe_camera/frame.hpp"

static std::atomic_bool shtdown; ///< Shutdown requested?

/// \brief Drive a LifecycleNode through configure -> activate via its
///        change_state service. Best-effort: logs and carries on when a
///        transition is rejected (e.g. the node was activated manually).
static void driveLifecycle(const rclcpp::Node::SharedPtr &node,
                           const std::string &target) {
  using lifecycle_msgs::msg::Transition;
  using lifecycle_msgs::srv::ChangeState;
  auto client = node->create_client<ChangeState>(target + "/change_state");
  if (!client->wait_for_service(std::chrono::seconds(10))) {
    LOG(WARNING) << target << "/change_state not available — is the node up?";
    return;
  }
  for (const std::uint8_t id :
       {Transition::TRANSITION_CONFIGURE, Transition::TRANSITION_ACTIVATE}) {
    auto request = std::make_shared<ChangeState::Request>();
    request->transition.id = id;
    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node, future,
                                           std::chrono::seconds(10)) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      LOG(WARNING) << target << ": lifecycle transition " << int(id)
                   << " timed out";
      return;
    }
    if (!future.get()->success) {
      LOG(WARNING) << target << ": lifecycle transition " << int(id)
                   << " rejected (already active?)";
    }
  }
  LOG(INFO) << target << " driven through configure + activate";
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0; // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> node =
      rclcpp::Node::make_shared("okvis_node_mowe");

  // Parameters.
  node->declare_parameter("config_filename", "");
  node->declare_parameter("imu_propagated_state_publishing_rate", 0.0);
  node->declare_parameter("camera_type", "arducam_ov9281");
  node->declare_parameter("camera_device", "/dev/video1");
  node->declare_parameter("camera_fps", 50.0);
  node->declare_parameter("activate_lifecycle_node", "/sch16t_imu_node");

  std::string configFilename;
  node->get_parameter("config_filename", configFilename);
  if (configFilename.empty()) {
    LOG(ERROR) << "ros parameter 'config_filename' not set";
    return EXIT_FAILURE;
  }
  double imu_propagated_state_publishing_rate = 0.0;
  node->get_parameter("imu_propagated_state_publishing_rate",
                      imu_propagated_state_publishing_rate);

  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);
  OKVIS_ASSERT_TRUE(std::runtime_error,
                    parameters.nCameraSystem.numCameras() == 2,
                    "okvis_node_mowe drives the OV9281 stereo pair — config "
                    "must declare exactly 2 cameras")

  // Publisher (pose, path, match visualisations).
  okvis::Publisher publisher(node);

  // DBoW2 vocabulary (the frontend loads it even when loop closures are off).
  boost::filesystem::path executable(argv[0]);
  const std::string dBowVocDir =
      executable.remove_filename().string() + "/../../share/okvis/resources/";
  std::ifstream infile(dBowVocDir + "/small_voc.yml.gz");
  if (!infile.good()) {
    LOG(ERROR) << "DBoW2 vocabulary " << dBowVocDir
               << "/small_voc.yml.gz not found.";
    return EXIT_FAILURE;
  }

  okvis::ThreadedSlam estimator(parameters, dBowVocDir);
  estimator.setBlocking(false);

  publisher.setBodyTransform(parameters.imu.T_BS);
  publisher.setOdometryPublishingRate(imu_propagated_state_publishing_rate);
  publisher.setupImageTopics(parameters.nCameraSystem);
  estimator.setOptimisedGraphCallback(
      std::bind(&okvis::Publisher::publishEstimatorUpdate, &publisher,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));

  // Bring the (lifecycle) IMU node up before subscribing — the launch-side
  // auto configure/activate helpers are unreliable.
  std::string lifecycleTarget;
  node->get_parameter("activate_lifecycle_node", lifecycleTarget);
  if (!lifecycleTarget.empty()) {
    driveLifecycle(node, lifecycleTarget);
  }

  // IMU in over ROS: sensor QoS (best-effort) with a deep queue, matching the
  // sch16t publisher; same conversion as okvis::Subscriber::imuCallback.
  auto imuSubscription = node->create_subscription<sensor_msgs::msg::Imu>(
      "imu0", rclcpp::SensorDataQoS().keep_last(1000),
      [&estimator, &publisher](const sensor_msgs::msg::Imu &msg) {
        const okvis::Time timestamp(msg.header.stamp.sec,
                                    msg.header.stamp.nanosec);
        const Eigen::Vector3d acc(msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z);
        const Eigen::Vector3d gyr(msg.angular_velocity.x,
                                  msg.angular_velocity.y,
                                  msg.angular_velocity.z);
        estimator.addImuMeasurement(timestamp, acc, gyr);
        publisher.realtimePredictAndPublish(timestamp, acc, gyr);
      });

  // Camera in directly via mowe_camera_core (no ROS on the image path).
  mowe::camera::CameraConfig cam_cfg;
  node->get_parameter("camera_type", cam_cfg.type);
  node->get_parameter("camera_device", cam_cfg.device);
  double fps = 50.0;
  node->get_parameter("camera_fps", fps);
  if (cam_cfg.type == "arducam_ov9281") {
    // Both eyes in one 2560x800 GREY frame; the driver splits to left/right
    // planes sharing one buffer + one SOF stamp (ADR-0005/ADR-0012).
    cam_cfg.width = 2560;
    cam_cfg.height = 800;
    cam_cfg.pixel_format = "GREY";
    cam_cfg.params["stereo"] = "true";
    cam_cfg.params["buffer_count"] = "4";
    cam_cfg.params["fps"] = std::to_string(fps);
  }

  try {
    auto camera = mowe::camera::CameraFactory::create(cam_cfg);
    if (camera->start() != mowe::camera::CaptureStatus::Ok) {
      LOG(ERROR) << "camera start failed (" << cam_cfg.type << " on "
                 << cam_cfg.device << ")";
      return EXIT_FAILURE;
    }
    LOG(INFO) << "camera streaming: " << cam_cfg.type << " on "
              << cam_cfg.device << " @ " << fps << " fps (direct ingest)";

    shtdown = false;
    signal(SIGINT, [](int) { shtdown = true; });

    // Capture thread: FrameBundle -> addImages. Planes are cloned: the
    // bundle's V4L2 buffer requeues when it goes out of scope, while the
    // estimator queue outlives it. (The GPU zero-copy path — NvBufSurface
    // straight into the XFeat engine — is a follow-up; ADR-0040 §3.)
    std::thread captureThread([&camera, &estimator]() {
      long timeouts = 0;
      while (!shtdown) {
        mowe::camera::FrameBundle bundle;
        if (camera->capture(std::chrono::milliseconds(200), bundle) !=
            mowe::camera::CaptureStatus::Ok) {
          if (++timeouts % 25 == 0) {
            LOG(WARNING) << "camera capture timeouts: " << timeouts;
          }
          continue;
        }
        const auto planes = bundle.planes();
        std::map<size_t, cv::Mat> images;
        for (const auto &plane : planes) {
          if (!plane.data)
            continue;
          const size_t camIdx = (plane.stream_id == "right") ? 1 : 0;
          images[camIdx] = cv::Mat(int(plane.height), int(plane.width), CV_8UC1,
                                   const_cast<std::uint8_t *>(plane.data),
                                   plane.stride_bytes)
                               .clone();
        }
        if (images.size() != 2) {
          continue; // need the full synchronized pair
        }
        okvis::Time t;
        t.fromNSec(std::uint64_t(bundle.timestamp().count()));
        estimator.addImages(t, images);
      }
      camera->stop();
    });

    // Main loop (same shape as okvis_node_subscriber).
    while (!shtdown) {
      rclcpp::spin_some(node);
      estimator.processFrame();
      std::map<std::string, cv::Mat> images;
      estimator.display(images);
      publisher.publishImages(images);
    }
    captureThread.join();

    if (parameters.estimator.do_final_ba) {
      std::cout << "final full BA..." << std::endl;
      estimator.doFinalBa();
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << e.what();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
