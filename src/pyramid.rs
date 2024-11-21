use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;
use std::fs;
use std::path::Path;

const EDGE_THRESHOLD: u32 = 19;

pub struct ImagePyramid {
    pub levels: Vec<GrayImage>,
    pub scale_factors: Vec<f32>,
    pub sigma_factors: Vec<f32>,
}

impl ImagePyramid {
    pub fn new(base_image: &GrayImage, n_levels: usize, scale_factor: f32) -> Self {
        let mut pyramid = ImagePyramid {
            levels: Vec::with_capacity(n_levels),
            scale_factors: Vec::with_capacity(n_levels),
            sigma_factors: Vec::with_capacity(n_levels),
        };
        pyramid.compute_pyramid(base_image, n_levels, scale_factor);
        pyramid
    }

    fn compute_pyramid(&mut self, base_image: &GrayImage, n_levels: usize, scale_factor: f32) {
        // Add padding to the base image once
        let padded_base = add_padding(base_image, EDGE_THRESHOLD);
        self.levels.push(padded_base.clone());
        self.scale_factors.push(1.0);
        self.sigma_factors.push(1.0);

        // Build the pyramid without adding padding at each level
        for i in 1..n_levels {
            let scale = scale_factor.powi(i as i32);
            self.scale_factors.push(scale);
            let sigma = scale_factor.sqrt();
            self.sigma_factors.push(sigma);

            // Use the unpadded base image for resizing
            let scaled = self.resize_image(&padded_base, scale);
            let blurred = gaussian_blur_f32(&scaled, sigma);

            // No padding is added here, only resizing and blurring
            self.levels.push(blurred);
        }
    }

    fn resize_image(&self, image: &GrayImage, scale: f32) -> GrayImage {
        let (width, height) = image.dimensions();
        let new_width = (width as f32 / scale) as u32;
        let new_height = (height as f32 / scale) as u32;

        ImageBuffer::from_fn(new_width, new_height, |x, y| {
            let src_x = (x as f32 * scale) as u32;
            let src_y = (y as f32 * scale) as u32;
            *image.get_pixel(src_x, src_y)
        })
    }

    pub fn save_levels(&self, output_dir: &str) -> Result<(), image::ImageError> {
        fs::create_dir_all(output_dir)?;
        for (i, level) in self.levels.iter().enumerate() {
            let filename = format!("level_{}.png", i);
            let path = Path::new(output_dir).join(filename);
            level.save(path)?;
        }
        Ok(())
    }
}

fn add_padding(image: &GrayImage, edge_threshold: u32) -> GrayImage {
    let (width, height) = image.dimensions();
    let new_width = width + 2 * edge_threshold;
    let new_height = height + 2 * edge_threshold;

    ImageBuffer::from_fn(new_width, new_height, |x, y| {
        if x < edge_threshold
            || y < edge_threshold
            || x >= width + edge_threshold
            || y >= height + edge_threshold
        {
            Luma([0]) // Padding pixel
        } else {
            *image.get_pixel(x - edge_threshold, y - edge_threshold)
        }
    })
}

//pub fn main() -> Result<(), Box<dyn std::error::Error>> {
//    let base_image = image::open("./test/image_1.jpeg")?.to_luma8();
//    let pyramid = ImagePyramid::new(&base_image, 8, 1.2);
//    pyramid.save_levels("./output/")?;
//    println!("Pyramid levels (both padded and unpadded) saved successfully!");
//    Ok(())
//}
