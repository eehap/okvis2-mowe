# okvis_xfeat_frontend — XFeat-on-TensorRT feature frontend (skeleton)

Bridges the Mow-e camera stack to OKVIS's feature frontend, replacing BRISK
detect+describe with **XFeat** running on **TensorRT**:

```
mowe::camera::FrameBundle            (mowe_camera_core, no ROS)
   │  per synchronized plane (left/right)
   ▼
map DMABUF / NvBufSurface → CUDA     (mowe_camera/gpu_map.hpp — zero-copy)
   ▼
CUDA preprocess (uint8 mono → padded NCHW float)        [TODO kernel]
   ▼
TensorRT XFeat engine (.plan) enqueue                   (TensorRTEngine)
   ▼
okvis::xfeat::FrameFeatures          → OKVIS MultiFrame adapter [next workstream]
```

This is **not** OKVIS's `USE_NN` path. That path is a Torch **semantic-segmentation**
model (Cityscapes sky/person classes — see `okvis_apps/src/nn_test.cpp`) used to
*mask* unreliable regions. XFeat is a separate, new feature frontend.

## Build

Opt-in from the top-level OKVIS build:

```bash
cmake -S okvis2-mowe -B build -DUSE_MOWE_XFEAT=ON \
      -Dmowe_camera_core_DIR=<prefix>/lib/cmake/mowe_camera_core
# real inference path (on the Jetson):
#   -DUSE_TENSORRT=ON   (in the submodule; needs CUDA 12.5 + TensorRT 10.3)
```

Or standalone:

```bash
cmake -S okvis2-mowe/okvis_xfeat_frontend -B build \
      -DCMAKE_PREFIX_PATH=<mowe_camera_core install prefix>
```

`mowe_camera_core` must be installed first (it exports a vanilla, ament-free
CMake package, so OKVIS finds it without ROS).

## Status — wired vs. stubbed

| Piece | State |
|---|---|
| `FrameBundle` → frontend plumbing, engine ownership, per-plane loop | ✅ real, compiles + runs |
| `mowe_camera::core` link + driver self-registration across the `.so` | ✅ verified on dev box |
| Zero-copy `NvBufSurface → EGL → CUDA` mapping (`gpu_map.cpp`) | ⚠️ written, **needs on-device verification** (JetPack 6.2 headers) |
| CUDA preprocess kernel (`preprocess.cu`, mono u8 → raw-0..255 NCHW float) | ⚠️ written, needs on-device build |
| TensorRT 10 binding introspection + `enqueueV3` (`TensorRTEngine.cpp`) | ⚠️ written, needs on-device build (verify TRT 10.3 API) |
| Device→host readback into `FrameFeatures` (score>0 filter) | ⚠️ written |
| `.plan` engine from `trtexec` | ❌ on-device step (see below) |
| OKVIS `MultiFrame` hand-off | ❌ next workstream (see below) |

Without `USE_TENSORRT` the engine runs in **stub mode**: the pipeline executes
end-to-end but emits empty features (useful for wiring/timing the capture path).

## The OKVIS hand-off (next workstream)

`FrameFeatures` carries **float** 64-D descriptors (cosine/L2 matching), not
binary BRISK (Hamming). Feeding OKVIS therefore requires the float-descriptor
matching path, not the BRISK one. The three mismatches to resolve when wiring
`FrameFeatures` → `okvis::MultiFrame`:

1. **Descriptor type/metric** — widen OKVIS keypoint/descriptor storage to float;
   replace Hamming scoring with L2/cosine.
2. **Matcher** — either reuse OKVIS's IMU-guided matching with the new metric, or
   bolt on **LighterGlue** (exported separately; see `../../xfeat_lightglue_onnx`).
3. **Place recognition** — OKVIS's DBoW2 vocabulary is BRISK-trained and unusable
   with XFeat descriptors; needs a new vocabulary or a different VPR path.

These are deliberately **out of scope** for this skeleton and warrant an ADR
(`docs/adr/`) per the repo's "new SLAM frontend / structural choice" rule.
