use image::{self, ImageBuffer, Rgb, RgbImage};
use nalgebra as na;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rust_vslam::{
    pose_estimator::CameraIntrinsics, tracker_types::SE3, visualization::VOVisualizer,
    vo::VisualOdometry,
};

const IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];
const SCALE_FACTOR: f64 = 1.0;

struct VOConfig {
    image_dir: String,
    output_dir: String,
    camera: Arc<CameraIntrinsics>,
}

// Implementation to be done

#[derive(Debug)]
enum ProcessError {
    IoError(std::io::Error),
    ImageError(image::ImageError),
    VOError(String),
    NoImagesFound,
}

impl From<std::io::Error> for ProcessError {
    fn from(err: std::io::Error) -> Self {
        ProcessError::IoError(err)
    }
}

impl From<image::ImageError> for ProcessError {
    fn from(err: image::ImageError) -> Self {
        ProcessError::ImageError(err)
    }
}

impl From<Box<dyn Error>> for ProcessError {
    fn from(err: Box<dyn Error>) -> Self {
        ProcessError::VOError(err.to_string())
    }
}

fn collect_image_paths(dir: &str) -> Result<Vec<PathBuf>, ProcessError> {
    let mut image_paths = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                if IMAGE_EXTENSIONS.contains(&ext_str.to_lowercase().as_str()) {
                    image_paths.push(path);
                }
            }
        }
    }

    if image_paths.is_empty() {
        return Err(ProcessError::NoImagesFound);
    }

    // Sort paths by filename
    image_paths.sort_by(|a, b| {
        let a_name = a.file_stem().and_then(|n| n.to_str()).unwrap_or("");
        let b_name = b.file_stem().and_then(|n| n.to_str()).unwrap_or("");
        a_name.cmp(b_name)
    });

    Ok(image_paths)
}

fn load_image(path: &Path) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, ProcessError> {
    let start = Instant::now();

    let image = image::open(path)?;

    // Scale image if needed
    let scaled = if SCALE_FACTOR != 1.0 {
        let width = (image.width() as f64 * SCALE_FACTOR) as u32;
        let height = (image.height() as f64 * SCALE_FACTOR) as u32;
        image.resize(width, height, image::imageops::FilterType::Lanczos3)
    } else {
        image
    };

    let rgb_image = scaled.to_rgb8();

    println!("Image loading took: {:?}", start.elapsed());
    Ok(rgb_image)
}

fn save_trajectory(file: &mut File, frame_id: usize, pose: &SE3) -> Result<(), ProcessError> {
    let rotation = na::UnitQuaternion::from_matrix(&pose.rotation);

    writeln!(
        file,
        "{} {} {} {} {} {} {} {}",
        frame_id,
        pose.translation[0],
        pose.translation[1],
        pose.translation[2],
        rotation.i,
        rotation.j,
        rotation.k,
        rotation.w,
    )?;

    Ok(())
}

fn process_sequence(config: &VOConfig) -> Result<(), ProcessError> {
    std::fs::create_dir_all(&config.output_dir)?;
    let matches_dir = Path::new(&config.output_dir).join("matches");
    std::fs::create_dir_all(&matches_dir)?;

    let mut trajectory_file = File::create(Path::new(&config.output_dir).join("trajectory.txt"))?;
    writeln!(trajectory_file, "# frame_id tx ty tz qx qy qz qw")?;

    let mut prev_image: Option<RgbImage> = None;

    let image_paths = collect_image_paths(&config.image_dir)?;
    println!("Found {} images to process", image_paths.len());

    // Maintain cumulative pose since vo.rs doesn't expose it
    let mut _cumulative_pose = SE3::identity();

    let visualizer = VOVisualizer::new();
    let mut vo = VisualOdometry::new(config.camera.clone(), 15.0);

    for (frame_idx, path) in image_paths.iter().enumerate() {
        println!(
            "\nProcessing frame {} : {:?}",
            frame_idx,
            path.file_name().unwrap()
        );

        let current_image = load_image(path)?;
        
        // Process frame normally...
        match vo.process_image(frame_idx, &current_image) {
            Ok(pose) => {
                save_trajectory(&mut trajectory_file, frame_idx, &pose)?;
                
                // Save match visualization every 10 frames
                if frame_idx % 10 == 0 && frame_idx > 0 {
                    if let Some(prev) = &prev_image {
                        let mut combined = RgbImage::new(
                            prev.width() + current_image.width(),
                            prev.height()
                        );
                        
                        // Copy images side by side
                        image::imageops::replace(&mut combined, prev, 0, 0);
                        image::imageops::replace(&mut combined, &current_image, prev.width() as i64, 0);
                        
                        // Save the visualization
                        let filename = format!("matches_{:04}.png", frame_idx);
                        combined.save(matches_dir.join(filename))?;
                    }
                }
            }
            Err(e) => eprintln!("Failed to process frame {}: {:?}", frame_idx, e),
        }

        prev_image = Some(current_image);
    }

    visualizer.save_trajectory_plot("output/trajectory_plot.png")?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let camera = Arc::new(CameraIntrinsics::new(
        1030.88 * SCALE_FACTOR,         // fx
        1069.57 * SCALE_FACTOR,         // fy
        353.15 * SCALE_FACTOR,          // cx
        564.86 * SCALE_FACTOR,          // cy
        (720.0 * SCALE_FACTOR) as u32, // width
        (1280.0 * SCALE_FACTOR) as u32,  // height
        0.0,
        0.0,
        0.0,
        0.0, // distortion parameters
    ));

    let config = VOConfig {
        image_dir: "test/outside".to_string(),
        output_dir: "poutput".to_string(),
        camera,
    };

    match process_sequence(&config) {
        Ok(_) => println!("Visual odometry processing completed successfully"),
        Err(e) => eprintln!("Error during processing: {:?}", e),
    }

    Ok(())
}
