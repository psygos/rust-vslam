use nalgebra as na;
use std::error::Error;

mod distributed_extraction;
use imageproc as ip;
use pyramid::ImagePyramid;
mod pyramid;

use image::{DynamicImage, GenericImageView, Luma, Rgb, RgbImage};
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_cross_mut;
use imageproc::point::Point;

fn main() -> Result<(), Box<dyn Error>> {
    let _ = distributed_extraction::main();
    Ok(())
}
