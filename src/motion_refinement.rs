use crate::pose_estimator::CameraIntrinsics;
use crate::tracker_types::*;
use nalgebra as na;
use std::sync::Arc;

const MAX_ITERATIONS: usize = 20;
const LAMBDA_INIT: f64 = 0.01;
const LAMBDA_FACTOR: f64 = 10.0;
const MIN_DELTA: f64 = 1e-6;
const MIN_DEPTH: f64 = 0.1;
const MAX_DEPTH: f64 = 40.0;

pub struct MotionOptimizer {
    camera: Arc<CameraIntrinsics>,
}

impl MotionOptimizer {
    pub fn new(camera: Arc<CameraIntrinsics>) -> Self {
        Self { camera }
    }

    pub fn refine_scale(
        &self,
        points_3d: &[na::Point3<f64>],
        prev_points_3d: &[na::Point3<f64>],
    ) -> f64 {
        let mut scales = Vec::new();
        for (p1, p2) in points_3d.iter().zip(prev_points_3d.iter()) {
            let dist = (p1 - p2).norm();
            if dist > 1e-6 {
                scales.push(dist);
            }
        }

        if scales.is_empty() {
            return 1.0;
        }

        scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        scales[scales.len() / 2] // Return median scale
    }

    pub fn refine_motion(
        &self,
        initial_pose: &SE3,
        points_3d: &[na::Point3<f64>],
        observations: &[na::Point2<f64>],
        weights: &[f64],
    ) -> TrackerResult<SE3> {
        println!("Entering refine_motion");

        if points_3d.len() != observations.len() || observations.len() != weights.len() {
            let err_msg = format!(
                "Inconsistent input sizes: points_3d={}, observations={}, weights={}",
                points_3d.len(),
                observations.len(),
                weights.len()
            );
            eprintln!("{}", err_msg);
            return Err(TrackerError::PoseEstimationFailed(err_msg));
        }

        println!("Input sizes consistent. Starting optimization.");

        let mut current_pose = initial_pose.clone();
        let mut lambda = LAMBDA_INIT;
        let mut last_error = self.compute_error(&current_pose, points_3d, observations, weights);

        println!("we got inside refine_motion");
        for _ in 0..MAX_ITERATIONS {
            let (jacobian, residuals) =
                self.compute_jacobian(&current_pose, points_3d, observations, weights);

            // Fix for moved value error: Clone jacobian before use
            let jacobian_t = jacobian.clone().transpose();
            let jtj = jacobian_t.clone() * &jacobian;
            let rhs = -(jacobian_t * residuals);

            let mut augmented_system = jtj;
            for i in 0..6 {
                augmented_system[(i, i)] += lambda;
            }

            match na::linalg::Cholesky::new(augmented_system) {
                Some(chol) => {
                    let delta = chol.solve(&rhs);
                    if delta.norm() < MIN_DELTA {
                        break;
                    }

                    let new_pose = self.update_pose(&current_pose, &delta);
                    let new_error = self.compute_error(&new_pose, points_3d, observations, weights);

                    if new_error < last_error {
                        current_pose = new_pose;
                        last_error = new_error;
                        lambda /= LAMBDA_FACTOR;
                    } else {
                        lambda *= LAMBDA_FACTOR;
                    }
                }
                None => {
                    lambda *= LAMBDA_FACTOR;
                    continue;
                }
            }
        }
        println!("we got here too");

        Ok(current_pose)
    }

    fn compute_jacobian(
        &self,
        pose: &SE3,
        points: &[na::Point3<f64>],
        observations: &[na::Point2<f64>],
        weights: &[f64],
    ) -> (na::DMatrix<f64>, na::DVector<f64>) {
        let n = points.len();
        let mut jacobian = na::DMatrix::zeros(2 * n, 6);
        let mut residuals = na::DVector::zeros(2 * n);

        for i in 0..n {
            let p = points[i];
            let obs = observations[i];
            let w = weights[i];

            // Transform point
            let transformed = pose.rotation * p + pose.translation;
            let z = transformed.z;

            if z.abs() < 1e-10 {
                continue;
            }

            let z_inv = 1.0 / z;
            let z_inv_sq = z_inv * z_inv;

            // Projected point
            let proj_x = self.camera.fx * transformed.x * z_inv + self.camera.cx;
            let proj_y = self.camera.fy * transformed.y * z_inv + self.camera.cy;

            // Residuals
            residuals[2 * i] = w * (proj_x - obs.x);
            residuals[2 * i + 1] = w * (proj_y - obs.y);

            // Jacobian blocks
            let x = transformed.x;
            let y = transformed.y;

            // For rotation (using Rodriguez formula derivatives)
            jacobian[(2 * i, 0)] = w * self.camera.fx * (y * z_inv);
            jacobian[(2 * i, 1)] = w * -self.camera.fx * (x * y * z_inv_sq);
            jacobian[(2 * i, 2)] = w * self.camera.fx * (-(1.0 + (x * x * z_inv_sq)));

            jacobian[(2 * i + 1, 0)] = w * -self.camera.fy * (x * z_inv);
            jacobian[(2 * i + 1, 1)] = w * self.camera.fy * (1.0 + (y * y * z_inv_sq));
            jacobian[(2 * i + 1, 2)] = w * -self.camera.fy * (x * y * z_inv_sq);

            // For translation
            jacobian[(2 * i, 3)] = w * self.camera.fx * z_inv;
            jacobian[(2 * i, 4)] = 0.0;
            jacobian[(2 * i, 5)] = w * -self.camera.fx * x * z_inv_sq;

            jacobian[(2 * i + 1, 3)] = 0.0;
            jacobian[(2 * i + 1, 4)] = w * self.camera.fy * z_inv;
            jacobian[(2 * i + 1, 5)] = w * -self.camera.fy * y * z_inv_sq;
        }

        (jacobian, residuals)
    }

    fn update_pose(&self, pose: &SE3, delta: &na::DVector<f64>) -> SE3 {
        let omega = na::Vector3::new(delta[0], delta[1], delta[2]);
        let theta = omega.norm();

        // Update rotation using Rodriguez formula
        let rotation = if theta < 1e-8 {
            pose.rotation.clone()
        } else {
            let omega_hat = na::Matrix3::new(
                0.0, -omega.z, omega.y, omega.z, 0.0, -omega.x, -omega.y, omega.x, 0.0,
            ) / theta;

            let rot_delta = na::Matrix3::identity()
                + omega_hat * theta.sin()
                + omega_hat * omega_hat * (1.0 - theta.cos());

            rot_delta * pose.rotation
        };

        let translation = pose.translation + na::Vector3::new(delta[3], delta[4], delta[5]);

        SE3 {
            rotation,
            translation,
        }
    }

    fn compute_error(
        &self,
        pose: &SE3,
        points: &[na::Point3<f64>],
        observations: &[na::Point2<f64>],
        weights: &[f64],
    ) -> f64 {
        let mut total_error = 0.0;
        let mut valid_points = 0;

        for i in 0..points.len() {
            let transformed = pose.rotation * points[i] + pose.translation;

            if transformed.z <= MIN_DEPTH || transformed.z >= MAX_DEPTH {
                continue;
            }

            let proj_x = self.camera.fx * transformed.x / transformed.z + self.camera.cx;
            let proj_y = self.camera.fy * transformed.y / transformed.z + self.camera.cy;

            let dx = proj_x - observations[i].x;
            let dy = proj_y - observations[i].y;
            total_error += weights[i] * (dx * dx + dy * dy);
            valid_points += 1;
        }

        if valid_points > 0 {
            total_error / valid_points as f64
        } else {
            f64::MAX
        }
    }
}
