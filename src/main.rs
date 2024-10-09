use image::{GrayImage, ImageBuffer};
use std::error::Error;
use std::path::Path;

pub struct ImagePyramid {
    levels: Vec<GrayImage>,
}

impl ImagePyramid {
    pub fn new(image: &GrayImage, num_levels: u32, scale_factor: f32) -> Self {
        let mut levels = Vec::with_capacity(num_levels as usize);
        levels.push(image.clone());

        for i in 1..num_levels {
            let previous = &levels[i as usize - 1];
            let scaled = Self::downsample(previous, scale_factor);
            levels.push(scaled);
        }

        ImagePyramid { levels }
    }

    fn downsample(image: &GrayImage, scale_factor: f32) -> GrayImage {
        let (width, height) = image.dimensions();

        let scaled_width = (width as f32 / scale_factor) as u32;
        let scaled_height = (height as f32 / scale_factor) as u32;

        // Write a new image
        ImageBuffer::from_fn(scaled_width, scaled_height, |x, y| {
            let scaled_x = (x as f32 * scale_factor) as u32;
            let scaled_y = (y as f32 * scale_factor) as u32;

            // Now just get the pixel value from original image given scaled coordinate
            *image.get_pixel(scaled_x, scaled_y)
        })
    }

    pub fn get_level(&self, level: usize) -> Option<&GrayImage> {
        self.levels.get(level)
    }

    pub fn save_levels(&self, output_dir: &str) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(output_dir)?;

        for (i, level) in self.levels.iter().enumerate() {
            let path = Path::new(output_dir).join(format!("level_{}.png", i));
            level.save(path)?;
        }

        Ok(())
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let test_img = image::open("./test/image_1.jpeg")?.to_luma8();
    let pyramid = ImagePyramid::new(&test_img, 4, 2.0);

    pyramid.save_levels("./output")?;

    for (i, level) in pyramid.levels.iter().enumerate() {
        println!("Level {}: {}x{}", i, level.width(), level.height());
    }

    Ok(())
}
