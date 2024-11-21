use image::{GrayImage, ImageBuffer, Luma};
use ocl::{Buffer, Device, Platform, ProQue};
use std::fs;
use std::path::Path;

const EDGE_THRESHOLD: u32 = 19;

const PYRAMID_KERNEL: &str = r#"
    __kernel void pyramid_level(
        __global const float* input,
        __global float* output,
        const int in_width,
        const int in_height,
        const int out_width,
        const int out_height,
        const float scale,
        const float sigma
    ) {
        const int x = get_global_id(0);
        const int y = get_global_id(1);
        
        if (x >= out_width || y >= out_height) return;
        
        // Gaussian kernel size (6 sigma rule)
        const int radius = (int)(3.0f * sigma);
        const float sigma2 = 2.0f * sigma * sigma;
        float sum = 0.0f;
        float value = 0.0f;
        
        // Source coordinates
        const float src_x = x * scale;
        const float src_y = y * scale;
        
        // Gaussian blur with proper scaling
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                const float gauss = exp(-(dx*dx + dy*dy) / sigma2);
                
                // Calculate source position with clamping
                const int sx = clamp((int)(src_x + dx), 0, in_width - 1);
                const int sy = clamp((int)(src_y + dy), 0, in_height - 1);
                
                const float pixel = input[sy * in_width + sx];
                value += pixel * gauss;
                sum += gauss;
            }
        }
        
        // Write normalized result
        output[y * out_width + x] = value / sum;
    }
"#;
#[derive(Clone)]

pub struct ImagePyramid {
    pub levels: Vec<GrayImage>,
    pub scale_factors: Vec<f32>,
    pub sigma_factors: Vec<f32>,
    pro_que: Option<ProQue>,
}

impl ImagePyramid {
    pub fn new(base_image: &GrayImage, n_levels: usize, scale_factor: f32) -> Self {
        let mut pyramid = ImagePyramid {
            levels: Vec::with_capacity(n_levels),
            scale_factors: Vec::with_capacity(n_levels),
            sigma_factors: Vec::with_capacity(n_levels),
            pro_que: None,
        };

        match pyramid.init_opencl() {
            Ok(()) => {
                pyramid.compute_pyramid(base_image, n_levels, scale_factor);
            }
            Err(e) => {
                eprintln!(
                    "Failed to initialize OpenCL: {}. Falling back to CPU implementation.",
                    e
                );
                pyramid.compute_pyramid_cpu(base_image, n_levels, scale_factor);
            }
        }

        pyramid
    }

    fn init_opencl(&mut self) -> ocl::Result<()> {
        let platform = Platform::default();
        let device = Device::first(platform)?;

        let pro_que = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(PYRAMID_KERNEL)
            .dims([1]) // Will be updated for each kernel execution
            .build()?;

        self.pro_que = Some(pro_que);
        Ok(())
    }

    fn compute_pyramid(&mut self, base_image: &GrayImage, n_levels: usize, scale_factor: f32) {
        let padded_base = add_padding(base_image, EDGE_THRESHOLD);
        self.levels.push(padded_base.clone());
        self.scale_factors.push(1.0);
        self.sigma_factors.push(1.0);

        let pro_que = self.pro_que.as_ref().unwrap();
        let input_data = convert_to_float_buffer(&padded_base);
        let input_dims = (padded_base.width(), padded_base.height());

        let input_buffer = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(input_dims.0 * input_dims.1)
            .copy_host_slice(&input_data)
            .build()
            .unwrap();

        for i in 1..n_levels {
            let scale = scale_factor.powi(i as i32);
            self.scale_factors.push(scale);
            let sigma = scale_factor.sqrt();
            self.sigma_factors.push(sigma);

            let new_width = (input_dims.0 as f32 / scale) as u32;
            let new_height = (input_dims.1 as f32 / scale) as u32;

            let output_buffer = Buffer::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::flags::MEM_WRITE_ONLY)
                .len(new_width * new_height)
                .build()
                .unwrap();

            let kernel = pro_que
                .kernel_builder("pyramid_level")
                .global_work_size([new_width, new_height])
                .arg(&input_buffer)
                .arg(&output_buffer)
                .arg(&(input_dims.0 as i32))
                .arg(&(input_dims.1 as i32))
                .arg(&(new_width as i32))
                .arg(&(new_height as i32))
                .arg(&scale)
                .arg(&sigma)
                .build()
                .unwrap();

            unsafe {
                kernel.enq().unwrap();
            }

            let mut result_buffer = vec![0.0f32; (new_width * new_height) as usize];
            output_buffer.read(&mut result_buffer).enq().unwrap();

            let level_image = convert_to_gray_image(&result_buffer, new_width, new_height);
            self.levels.push(level_image);
        }
    }

    fn compute_pyramid_cpu(&mut self, base_image: &GrayImage, n_levels: usize, scale_factor: f32) {
        let padded_base = add_padding(base_image, EDGE_THRESHOLD);
        self.levels.push(padded_base.clone());
        self.scale_factors.push(1.0);
        self.sigma_factors.push(1.0);

        for i in 1..n_levels {
            let scale = scale_factor.powi(i as i32);
            self.scale_factors.push(scale);
            let sigma = scale_factor.sqrt();
            self.sigma_factors.push(sigma);

            println!("Processing level {} (CPU)", i);
            let scaled = self.resize_image(&padded_base, scale);
            let blurred = imageproc::filter::gaussian_blur_f32(&scaled, sigma);
            self.levels.push(blurred);
        }
    }

    fn resize_image(&self, image: &GrayImage, scale: f32) -> GrayImage {
        let (width, height) = image.dimensions();
        let new_width = (width as f32 / scale) as u32;
        let new_height = (height as f32 / scale) as u32;

        image::imageops::resize(
            image,
            new_width,
            new_height,
            image::imageops::FilterType::Lanczos3,
        )
    }

    pub fn save_levels(&self, output_dir: &str) -> Result<(), image::ImageError> {
        fs::create_dir_all(output_dir)?;
        for (i, level) in self.levels.iter().enumerate() {
            let filename = format!("level_{}.png", i);
            let path = Path::new(output_dir).join(filename);
            let path_str = path.display().to_string();
            level.save(&path)?;
            println!("Saved pyramid level {} to {}", i, path_str);
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
            Luma([0])
        } else {
            *image.get_pixel(x - edge_threshold, y - edge_threshold)
        }
    })
}

fn convert_to_float_buffer(image: &GrayImage) -> Vec<f32> {
    image.pixels().map(|p| p[0] as f32 / 255.0).collect()
}

fn convert_to_gray_image(buffer: &[f32], width: u32, height: u32) -> GrayImage {
    ImageBuffer::from_fn(width, height, |x, y| {
        let idx = (y * width + x) as usize;
        let value = (buffer[idx] * 255.0).clamp(0.0, 255.0) as u8;
        Luma([value])
    })
}
