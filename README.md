# Rust Visual SLAM v0.1.0 🚀

A fast, memory-safe Visual Odometry implementation in Rust with GPU-accelerated image processing.

![Feature Matching](docs/images/feature_matching.png)
![Trajectory](docs/images/trajectory.png)

## Features

- Feature detection and tracking using FAST corners with ORB descriptors
- GPU-accelerated image pyramid generation
- Hybrid pose estimation (PnP + Essential Matrix)
- Local mapping with keyframe management
- Real-time trajectory visualization
- Quadtree-based feature distribution for robust tracking

## Quick Start

### Prerequisites

- Rust 1.70+
- OpenCL-capable GPU (optional, falls back to CPU)
- ffmpeg (for video preprocessing)

### Installation

```bash
git clone https://github.com/yourusername/rust-vslam
cd rust-vslam
cargo build --release
```

### Converting Video to Image Sequence

```bash
# Create image sequence directory
mkdir -p test/outside

# Convert video to image sequence (30 fps)
ffmpeg -i your_video.mp4 -vf fps=30 test/outside/%04d.jpg
```

### Camera Calibration

Obtain your camera parameters using either:
- OpenCV camera calibration
- [Online Camera Calibrator](https://onlinecameracalibration.com/)

Update `main.rs` with your camera parameters:
```rust
let camera = Arc::new(CameraIntrinsics::new(
    fx,    // focal length x
    fy,    // focal length y
    cx,    // principal point x
    cy,    // principal point y
    width, // image width
    height // image height
));
```

### Running

```bash
cargo run --release
```

Output files will be generated in `output/`:
- `trajectory_plot.png`: Top-down view of camera motion
- `matches_*.png`: Feature matching visualizations

## Immediate Roadmap

- [ ] Loop closure detection and correction
- [ ] Local bundle adjustment
- [ ] TUM dataset compatibility
- [ ] Improved scale consistency
- [ ] Bidirectional feature matching

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
