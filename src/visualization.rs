use image::Rgb;
use imageproc::drawing::{draw_cross_mut, draw_line_segment_mut};
use kiss3d::light::Light;
use kiss3d::nalgebra as na;
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
use plotters::prelude::*;
use std::collections::VecDeque;

const TRAJECTORY_LENGTH: usize = 100; // Keep last N poses for visualization
const CAMERA_SIZE: f32 = 0.1;
const POINT_SIZE: f32 = 0.02;
const LINE_RADIUS: f32 = 0.01;

pub struct VOVisualizer {
    window: Window,
    camera_trail: VecDeque<(na::Point3<f32>, na::UnitQuaternion<f32>)>,
    camera_model: SceneNode,
    trail_points: Vec<SceneNode>,
}

impl VOVisualizer {
    pub fn new() -> Self {
        let mut window = Window::new("Visual Odometry Trajectory");
        window.set_background_color(0.1, 0.1, 0.1);
        window.set_light(Light::StickToCamera);

        // Create camera model
        let mut camera_model = window.add_group();

        // Camera frustum
        let mut body = camera_model.add_cube(CAMERA_SIZE, CAMERA_SIZE / 2.0, CAMERA_SIZE / 3.0);
        body.set_color(0.2, 0.8, 0.2);

        // Add coordinate axes to show orientation
        let mut x_axis = camera_model.add_cylinder(LINE_RADIUS, CAMERA_SIZE / 2.0);
        x_axis.set_color(1.0, 0.0, 0.0);
        x_axis.append_rotation(&na::UnitQuaternion::from_euler_angles(
            0.0,
            0.0,
            std::f32::consts::FRAC_PI_2,
        ));

        let mut y_axis = camera_model.add_cylinder(LINE_RADIUS, CAMERA_SIZE / 2.0);
        y_axis.set_color(0.0, 1.0, 0.0);

        let mut z_axis = camera_model.add_cylinder(LINE_RADIUS, CAMERA_SIZE / 2.0);
        z_axis.set_color(0.0, 0.0, 1.0);
        z_axis.append_rotation(&na::UnitQuaternion::from_euler_angles(
            std::f32::consts::FRAC_PI_2,
            0.0,
            0.0,
        ));

        Self {
            window,
            camera_trail: VecDeque::with_capacity(TRAJECTORY_LENGTH),
            camera_model,
            trail_points: Vec::new(),
        }
    }

    pub fn update(&mut self, r: &na::Matrix3<f64>, t: &na::Vector3<f64>) -> bool {
        // Convert double precision to single precision for visualization
        let position = na::Point3::new(t[0] as f32, t[1] as f32, t[2] as f32);
        let rotation = na::UnitQuaternion::from_matrix(&na::Matrix3::new(
            r[(0, 0)] as f32,
            r[(0, 1)] as f32,
            r[(0, 2)] as f32,
            r[(1, 0)] as f32,
            r[(1, 1)] as f32,
            r[(1, 2)] as f32,
            r[(2, 0)] as f32,
            r[(2, 1)] as f32,
            r[(2, 2)] as f32,
        ));

        // Update camera trail
        if self.camera_trail.len() >= TRAJECTORY_LENGTH {
            self.camera_trail.pop_front();
        }
        self.camera_trail.push_back((position, rotation));

        // Update camera position and orientation
        self.camera_model
            .set_local_translation(na::Translation3::from(position.coords));
        self.camera_model.set_local_rotation(rotation);

        // Clean up old visualization elements
        for point in &mut self.trail_points {
            self.window.remove_node(point);
        }
        self.trail_points.clear();

        // Draw trail points
        for (pos, _) in &self.camera_trail {
            let mut point = self.window.add_sphere(POINT_SIZE);
            point.set_color(0.8, 0.8, 0.2);
            point.set_local_translation(na::Translation3::from(pos.coords));
            self.trail_points.push(point);
        }

        // Draw connecting lines between trail points using thin cylinders
        if self.camera_trail.len() >= 2 {
            for i in 0..self.camera_trail.len() - 1 {
                let (p1, _) = self.camera_trail[i];
                let (p2, _) = self.camera_trail[i + 1];

                // Calculate cylinder properties for line segment
                let direction = p2 - p1;
                let length = direction.norm();
                let rotation = if length > 1e-6 {
                    na::UnitQuaternion::rotation_between(
                        &na::Vector3::new(0.0, 1.0, 0.0),
                        &(direction / length),
                    )
                    .unwrap_or(na::UnitQuaternion::identity())
                } else {
                    na::UnitQuaternion::identity()
                };

                let midpoint = p1 + (direction / 2.0);
                let mut cylinder = self.window.add_cylinder(LINE_RADIUS, length);
                cylinder.set_color(0.5, 0.5, 0.8);
                cylinder.set_local_translation(na::Translation3::from(midpoint.coords));
                cylinder.set_local_rotation(rotation);
                self.trail_points.push(cylinder);
            }
        }

        self.window.render()
    }

    pub fn save_trajectory_plot(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let x_values: Vec<f32> = self.camera_trail.iter().map(|(pos, _)| pos.x).collect();
        let z_values: Vec<f32> = self.camera_trail.iter().map(|(pos, _)| pos.z).collect();

        let min_x = x_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_x = x_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_z = z_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_z = z_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption("Camera Trajectory (Top View)", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d((min_x - 1.0)..(max_x + 1.0), (min_z - 1.0)..(max_z + 1.0))?;

        chart.configure_mesh().draw()?;

        // Draw trajectory
        chart.draw_series(LineSeries::new(
            x_values.iter().zip(z_values.iter()).map(|(&x, &z)| (x, z)),
            &BLUE,
        ))?;

        // Mark start and end points
        if let Some((start_pos, _)) = self.camera_trail.front() {
            chart.draw_series(PointSeries::of_element(
                vec![(start_pos.x, start_pos.z)],
                5,
                &GREEN,
                &|coord, size, style| {
                    EmptyElement::at(coord)
                        + Circle::new((0, 0), size, style.filled())
                        + Text::new("Start", (10, 0), ("sans-serif", 15))
                },
            ))?;
        }

        if let Some((end_pos, _)) = self.camera_trail.back() {
            chart.draw_series(PointSeries::of_element(
                vec![(end_pos.x, end_pos.z)],
                5,
                &RED,
                &|coord, size, style| {
                    EmptyElement::at(coord)
                        + Circle::new((0, 0), size, style.filled())
                        + Text::new("Current", (10, 0), ("sans-serif", 15))
                },
            ))?;
        }

        Ok(())
    }
}

pub fn visualize_matches(
    frame1: &image::RgbImage,
    frame2: &image::RgbImage,
    matches: &[crate::feature_matcher::FeatureMatch],
    descriptors1: &[Vec<crate::new_distributed_extraction::OrbDescriptor>],
    descriptors2: &[Vec<crate::new_distributed_extraction::OrbDescriptor>],
    inliers: &[bool],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w1, h1) = frame1.dimensions();
    let (w2, h2) = frame2.dimensions();

    let mut combined = image::RgbImage::new(w1 + w2, h1.max(h2));

    // Copy original images
    image::imageops::replace(&mut combined, frame1, 0i64, 0i64);
    image::imageops::replace(&mut combined, frame2, w1 as i64, 0i64);

    // Draw matches
    for (&is_inlier, m) in inliers.iter().zip(matches.iter()) {
        let kp1 = &descriptors1[m.query_level][m.query_idx].keypoint;
        let kp2 = &descriptors2[m.train_level][m.train_idx].keypoint;

        let x1 = kp1.pt.0 as i32;
        let y1 = kp1.pt.1 as i32;
        let x2 = kp2.pt.0 as i32 + w1 as i32;
        let y2 = kp2.pt.1 as i32;

        let color = if is_inlier {
            Rgb([0, 255, 0]) // Green for inliers
        } else {
            Rgb([255, 0, 0]) // Red for outliers
        };

        draw_cross_mut(&mut combined, color, x1, y1);
        draw_cross_mut(&mut combined, color, x2, y2);
        draw_line_segment_mut(
            &mut combined,
            (x1 as f32, y1 as f32),
            (x2 as f32, y2 as f32),
            color,
        );
    }

    combined.save(output_path)?;
    Ok(())
}
