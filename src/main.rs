use nalgebra as na;
use std::error::Error;

use imageproc as ip;
use pyramid::ImagePyramid;
mod pyramid;

use image::{DynamicImage, GenericImageView, Luma, Rgb, RgbImage};
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_cross_mut;
use imageproc::point::Point;

fn main() -> Result<(), Box<dyn Error>> {
    //let test_img = image::open("../test/image_1.jpeg")?.to_luma8();
    //let pyramid = ImagePyramid::new(&test_img, 4, 2.0);

    //pyramid.save_levels("../output")?;

    //for (i, level) in pyramid.levels.iter().enumerate() {
    //    println!("Level {}: {}x{}", i, level.width(), level.height());
    //}
    let gray_image = image::open("../test/fast_test.jpeg")?.to_luma8();

    let threshold = 30;
    let corners = corners_fast9(&gray_image, threshold);

    let dynamic_image = DynamicImage::ImageLuma8(gray_image);
    let mut rgb_image = dynamic_image.to_rgb8();

    for corner in &corners {
        draw_cross_mut(
            &mut rgb_image,
            Rgb([255, 0, 0]),
            corner.x as i32,
            corner.y as i32,
        );
    }

    rgb_image.save("../output/fast_test_corners.jpg")?;

    println!(
        "Detected {} corners. Output saved as ../output/fast_test_corners.jpg",
        corners.len()
    );

    Ok(())
}
