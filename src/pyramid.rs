use image::{GrayImage, ImageBuffer};
use imageproc::filter::gaussian_blur_f32;
use std::fs;
use std::path::Path;

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
        self.levels.push(base_image.clone());
        self.scale_factors.push(1.0);
        self.sigma_factors.push(1.0);

        for i in 1..n_levels {
            let scale = scale_factor.powi(i as i32);
            self.scale_factors.push(scale);

            let sigma = scale_factor.sqrt();
            self.sigma_factors.push(sigma);

            let previous = &self.levels[i - 1];
            let scaled = self.resize_image(previous, scale);
            let blurred = gaussian_blur_f32(&scaled, sigma);
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

    pub fn get_level(&self, level: usize) -> Option<&GrayImage> {
        self.levels.get(level)
    }

    pub fn get_scale_factor(&self, level: usize) -> Option<f32> {
        self.scale_factors.get(level).cloned()
    }

    pub fn get_sigma_factor(&self, level: usize) -> Option<f32> {
        self.sigma_factors.get(level).cloned()
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }
    pub fn save_levels(&self, output_dir: &str) -> Result<(), image::ImageError> {
        fs::create_dir_all(output_dir)?;

        for (i, level) in self.levels.iter().enumerate() {
            let filename = format!("new_level_{}.png", i);
            let path = Path::new(output_dir).join(filename);
            level.save(path)?;
        }

        Ok(())
    }
}

// Example usage
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_image = image::open("./test/image_1.jpeg").unwrap().to_luma8();
    let pyramid = ImagePyramid::new(&base_image, 8, 1.2);

    pyramid.save_levels("./output/")?;

    println!("Pyramid levels saved successfully!");
    Ok(())
}
