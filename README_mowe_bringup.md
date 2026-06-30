# OKVIS2 bring-up on the Mow-e bench rig (OV9281 stereo + SCH16T IMU)

Stand up **visual-inertial odometry** on the bench using stock OKVIS2 fed by the
`mowe_camera` (OV9281 global-shutter stereo) and `sch16t_imu_node` (Murata
SCH16T) ROS 2 nodes, and publish a pose.

This is the **BRISK-first** path: stock OKVIS feature frontend (BRISK
detect/describe + IMU-guided matching). It de-risks IMU ingestion, camera/IMU
time alignment, and pose output **before** the XFeat + LighterGlue frontend swap
(ADR-0040 issues #4–6). Once this loop is closed and stable, the frontend swap
is a separate branch — nothing here is throwaway (config, extrinsics, IMU model,
and the QoS/time-sync work all carry straight over).

## Files this adds

| File | Purpose |
|---|---|
| `config/ov9281_sch16t.yaml` | OKVIS config: stereo intrinsics/extrinsics + SCH16T IMU model. **Has calibration TODOs — read them.** |
| `launch/okvis_mowe_ov9281.launch.xml` | Brings up camera + IMU (via their own lifecycle launches) + OKVIS, with topic remaps. |
| `okvis_ros2/src/Subscriber.cpp` (patched) | Subscribes **best-effort** to match the sensors' `SensorDataQoS`. |

## ⚠️ Before the pose is meaningful: calibration

OKVIS needs camera intrinsics and camera↔IMU extrinsics. Status on the bench:

1. **Camera intrinsics** (`focal_length`, `principal_point`, `distortion`) —
   **REQUIRED, and you can do this now.** Single-camera checkerboard:
   `ros2 run camera_calibration cameracalibrator` (or `cv2.calibrateCamera`).
   The values in the config are sensor-geometry **placeholders** (`f≈1000 px`,
   principal point = image centre, zero distortion) — structure/scale will be
   wrong until replaced. Tag: `# TODO-CALIB`.
2. **Stereo extrinsics** (cam0↔cam1 baseline, in each `T_SC`) — placeholder
   0.10 m baseline. Set from mechanical CAD now; refine with Kalibr later.
3. **Camera↔IMU extrinsics** (`T_SC` rotation/translation vs the SCH16T) —
   the "no means yet" TODO. Placeholder = axis-aligned mount. Needs **Kalibr**
   (`kalibr_calibrate_imu_camera`) eventually; until then VIO runs but the lever
   arm/orientation bias degrades accuracy.
4. **IMU noise model** (`sigma_*` in `imu_parameters`) — **done from the
   SCH16T-K01 datasheet** (Doc.No. 11624 Rev.6, Tables 4 & 6). White-noise
   densities and saturation/offset are datasheet-direct; the bias *random-walk*
   densities (`sigma_gw_c`, `sigma_aw_c`) are scaled estimates — the datasheet
   gives bias *instability*, not the +½-slope RW term — and want an on-rig
   Allan-variance refinement. If the filter is over-confident/diverges, inflate
   the white-noise densities 2–4×.

VIO will *run* with the placeholders — useful to confirm data flow and the time
sync — but don't trust the trajectory until at least (1) and a real (2) are in.

## Timestamps — camera ↔ IMU (the dual-stamp question)

The usual VIO nightmare is two sensors on **different clock epochs**. We avoid
it: **both nodes already stamp on `CLOCK_MONOTONIC`** —

- `sch16t_node.cpp` → `rclcpp::Time(..., RCL_STEADY_TIME)`, captured at SPI-read
  entry.
- `ov9281_sync_ros_node.cpp` → `bundle.timestamp()` = monotonic **SOF**
  (start-of-frame), per ADR-0015.

So there is **one clock epoch, no host/sensor skew**. Two residuals remain, both
small and handled in config, not code:

- **Sub-frame offset.** The camera stamp is start-of-frame (not mid-exposure)
  and the IMU stamp carries DRDY→SPI-read latency. Absorbed by
  `camera_parameters: image_delay` (start at `0.0`; a slightly negative value
  pulls camera time toward mid-exposure ≈ −exposure/2). This is also exactly
  what a camera↔IMU time-offset calibration would refine.
- **Clock-type label mismatch (cosmetic but a tooling trap).** The IMU builds
  its `rclcpp::Time` with `RCL_STEADY_TIME`; the camera builds it from a raw ns
  count, so it's *tagged* `RCL_ROS_TIME` even though the value is monotonic ns.
  OKVIS reads `header.stamp.sec/.nanosec` raw and ignores the tag, so they fuse
  correctly — but any node that compares `rclcpp::Time` objects across the two
  (or uses TF with `use_sim_time`) will trip on the label. If that bites later,
  stamp the camera with `RCL_STEADY_TIME` too.

The stereo pair itself is hardware-synced by `ov9281_sync` and emits **identical
stamps** (split mode), so `timestamp_tolerance` is tight (3 ms).

## QoS (why the Subscriber was patched)

`mowe_camera` and `sch16t_imu_node` publish `rclcpp::SensorDataQoS()` →
**BEST_EFFORT**. Stock OKVIS subscribed with the default **RELIABLE** profile;
a reliable subscriber is **QoS-incompatible** with a best-effort publisher and
silently receives **nothing**. `Subscriber.cpp` now subscribes best-effort
(images via `rmw_qos_profile_sensor_data`, IMU via
`rclcpp::SensorDataQoS().keep_last(1000)`). A best-effort sub still accepts a
reliable publisher, so rosbag playback keeps working.

## Build

```bash
cd ~/<ros2_ws>
colcon build --packages-select mowe_camera sch16t_imu_node okvis
source install/setup.bash
```

The OKVIS DBoW2 vocabulary (`resources/small_voc.yml.gz`) must be installed — it
ships with the `okvis` package. BRISK frontend uses it for loop closure.

## Run

```bash
ros2 launch okvis okvis_mowe_ov9281.launch.xml
```

Useful args:

| Arg | Default | Notes |
|---|---|---|
| `config_filename` | `…/okvis/config/ov9281_sch16t.yaml` | OKVIS config |
| `camera` / `imu` | `true` | set `false` to feed that stream from a rosbag instead |
| `rviz` | `true` | OKVIS RViz config (trajectory + keyframes) |

### Sanity checks (in order)

```bash
# 1. Are the sensors actually publishing best-effort streams?
ros2 topic hz /mowe_camera/left/image_raw  --qos-reliability best_effort
ros2 topic hz /mowe_camera/right/image_raw --qos-reliability best_effort
ros2 topic hz /imu/data_raw                --qos-reliability best_effort

# 2. Is OKVIS emitting a pose? (publisher rate set to 200 Hz in the launch)
#    Topics (node is namespaced /okvis): okvis_odometry (nav_msgs/Odometry),
#    okvis_transform (geometry_msgs/TransformStamped), okvis_path (Marker).
ros2 topic hz   /okvis/okvis_odometry
ros2 topic echo /okvis/okvis_transform --once
```

If OKVIS logs `Not enough keypoints (<15)` repeatedly, the intrinsics/exposure
are off (placeholders) or the scene is texture-poor — fix intrinsics first.

## What "working" looks like

- A live pose updates at the IMU-propagation rate, drifting slowly when static
  (expected before calibration), tracking when the rig moves.
- RViz shows a growing keyframe trajectory; loop closures snap it on revisit.
- The trajectory is the right **shape**; correct **scale** waits on real
  intrinsics + a measured baseline.

## Next branch — XFeat + LighterGlue frontend swap

Once the pose loop is solid, swap BRISK for the XFeat-on-TensorRT frontend
(ADR-0040). That is **not** a config change — `ThreadedSlam` runs
`Frontend::detectAndDescribe` (BRISK) internally; the swap means float-descriptor
`MultiFrame` storage, L2/cosine matching (or LighterGlue), and a non-BRISK VPR.
Tracked as ADR-0040 issues #4–6.
