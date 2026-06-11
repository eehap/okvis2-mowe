/**
 * @file xfeat_frontend_demo.cpp
 * @brief End-to-end skeleton: open a mowe_camera, capture frames, run the XFeat
 *        frontend, print feature counts. Proves the FrameBundle → frontend wire
 *        without ROS. (Inference itself is a stub until an engine is supplied.)
 *
 * Usage: xfeat_frontend_demo <camera_type> <device> [engine.plan]
 *   e.g. xfeat_frontend_demo arducam_ov9281 /dev/video0 xfeat.plan
 */
#include <chrono>
#include <iostream>
#include <string>

#include "mowe_camera/camera.hpp"
#include "mowe_camera/factory.hpp"
#include "mowe_camera/frame.hpp"

#include "okvis/xfeat/XFeatFrontend.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <camera_type> <device> [engine.plan]\n";
    return 1;
  }

  mowe::camera::CameraConfig cam_cfg;
  cam_cfg.type = argv[1];
  cam_cfg.device = argv[2];
  // OV9281 jetvariety stereo: both eyes arrive as one 2560x800 mono frame; the
  // driver splits to left/right planes (1280x800 each) in place — matches the
  // per-eye XFeat engine. See mowe-camera/config/ov9281_stereo.yaml (ADR-0005).
  if (cam_cfg.type == "arducam_ov9281") {
    cam_cfg.width = 2560;
    cam_cfg.height = 800;
    cam_cfg.pixel_format = "GREY";
    cam_cfg.params["stereo"] = "true";
    cam_cfg.params["buffer_count"] = "4";
    cam_cfg.params["fps"] = "50.0";
  }

  okvis::xfeat::XFeatConfig fe_cfg;  // input dims auto-read from the engine
  if (argc >= 4) fe_cfg.engine_path = argv[3];

  try {
    auto camera = mowe::camera::CameraFactory::create(cam_cfg);
    okvis::xfeat::XFeatFrontend frontend(fe_cfg);

    if (camera->start() != mowe::camera::CaptureStatus::Ok) {
      std::cerr << "camera start failed\n";
      return 2;
    }

    for (int i = 0; i < 100; ++i) {
      mowe::camera::FrameBundle bundle;
      if (camera->capture(std::chrono::milliseconds(200), bundle) !=
          mowe::camera::CaptureStatus::Ok) {
        std::cerr << "capture timeout/error\n";
        continue;
      }
      auto feats = frontend.extract(bundle);
      std::cout << "seq " << feats.sequence << " t " << feats.timestamp_ns
                << "ns:";
      for (const auto& s : feats.streams) {
        std::cout << " [" << s.stream_id << "] " << s.size() << " kpts";
      }
      std::cout << (frontend.engine_loaded() ? "" : "  (stub)") << "\n";
    }
    camera->stop();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
