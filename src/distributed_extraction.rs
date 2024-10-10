use crate::pyramid::*;
use image::{GenericImageView, GrayImage};
use image::{Rgb, RgbImage};
use imageproc::corners::{corners_fast9, Corner};
use imageproc::drawing::draw_cross_mut;

const EDGE_THRESHOLD: i32 = 19;

pub struct KeyPoint {
    pub pt: (f32, f32),
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

pub fn compute_keypoints_oct_tree(
    pyramid: &ImagePyramid,
    ini_th_fast: u8,
    min_th_fast: u8,
    max_keypoints: usize,
) -> Vec<Vec<KeyPoint>> {
    let mut all_keypoints = Vec::new();
    let w: i32 = 30; // Cell size

    for (level, image) in pyramid.levels.iter().enumerate() {
        let (width, height) = image.dimensions();
        let mut keypoints = Vec::new();

        let min_border_x = EDGE_THRESHOLD - 3;
        let max_border_x = width as i32 - EDGE_THRESHOLD + 3;
        let min_border_y = min_border_x;
        let max_border_y = height as i32 - EDGE_THRESHOLD + 3;

        let width_range = max_border_x - min_border_x;
        let height_range = max_border_y - min_border_y;

        let n_cols = width_range / w;
        let n_rows = height_range / w;

        let w_cell = ((width_range as f32 / n_cols as f32).ceil()) as i32;
        let h_cell = ((height_range as f32 / n_rows as f32).ceil()) as i32;

        for i in 0..n_rows {
            let ini_y = min_border_y + i * h_cell;
            let max_y = ini_y + h_cell + 6;

            if ini_y >= max_border_y - 3 {
                continue;
            }

            for j in 0..n_cols {
                let ini_x = min_border_x + j * w_cell;
                let max_x = ini_x + w_cell + 6;

                if ini_x >= max_border_x - 6 {
                    continue;
                }

                let cell_keypoints = detect_keypoints_in_cell(
                    image,
                    ini_x.max(0) as u32,
                    ini_y.max(0) as u32,
                    max_x.min(width as i32) as u32,
                    max_y.min(height as i32) as u32,
                    ini_th_fast,
                    min_th_fast,
                );

                for mut kp in cell_keypoints {
                    kp.pt.0 += j as f32 * w_cell as f32;
                    kp.pt.1 += i as f32 * h_cell as f32;
                    kp.octave = level as i32;
                    keypoints.push(kp);
                }
            }
        }

        all_keypoints.push(keypoints);
    }

    all_keypoints
}

fn detect_keypoints_in_cell(
    image: &GrayImage,
    ini_x: u32,
    ini_y: u32,
    max_x: u32,
    max_y: u32,
    ini_th_fast: u8,
    min_th_fast: u8,
) -> Vec<KeyPoint> {
    let sub_image = image.view(ini_x, ini_y, max_x - ini_x, max_y - ini_y);
    let cell_image = sub_image.to_image();
    let mut corners = corners_fast9(&cell_image, ini_th_fast);

    if corners.is_empty() {
        corners = corners_fast9(&cell_image, min_th_fast);
    }

    corners
        .into_iter()
        .map(|corner| KeyPoint {
            pt: (corner.x as f32, corner.y as f32),
            size: 7.0,   // typical size for FAST features
            angle: -1.0, // to be computed later
            response: corner.score as f32,
            octave: 0, // to be set based on the pyramid level
        })
        .collect()
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("./test/fast_test.jpeg")?;
    let gray_img = img.to_luma8();

    let pyramid = ImagePyramid::new(&gray_img, 8, 1.2);

    let all_keypoints = compute_keypoints_oct_tree(&pyramid, 20, 7, 1000);

    let mut colored_img = img.to_rgb8();

    // mark keypoints of each level differently
    let colors = [
        Rgb([255, 0, 0]),     // Red
        Rgb([0, 255, 0]),     // Green
        Rgb([0, 0, 255]),     // Blue
        Rgb([255, 255, 0]),   // Yellow
        Rgb([255, 0, 255]),   // Magenta
        Rgb([0, 255, 255]),   // Cyan
        Rgb([128, 128, 128]), // Gray
        Rgb([255, 128, 0]),   // Orange
    ];

    for (level, keypoints) in all_keypoints.iter().enumerate() {
        let color = colors[level % colors.len()];
        let scale_factor = pyramid.scale_factors[level];
        for kp in keypoints {
            let x = kp.pt.0.round() as i32 * scale_factor as i32;
            let y = kp.pt.1.round() as i32 * scale_factor as i32;

            draw_cross_mut(&mut colored_img, color, x, y);
        }
    }

    colored_img.save("./output/keypoints_visualization.png")?;

    println!("Keypoint visualization saved as 'keypoints_visualization.png'");

    Ok(())
}
