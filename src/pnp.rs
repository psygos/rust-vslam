use crate::tracker_types::{TrackerError, TrackerResult, SE3};
use nalgebra as na;
use rand::prelude::*;
/// THIS IS A PLACEHOLDER LLMs GENERATED CODE THAT NEEDS TO BE IMPROVED
/// Constants for PnP solver 
const MIN_POINTS: usize = 4;
const P3P_POINTS: usize = 3;
const MIN_PARALLAX: f64 = 1e-6;
const MIN_DEPTH: f64 = 0.1;
const MAX_DEPTH: f64 = 40.0;
const LM_MAX_ITERS: usize = 10;
const LM_EPSILON: f64 = 1e-7;
const INIT_LAMBDA: f64 = 1e-3;
const LAMBDA_FACTOR: f64 = 10.0;

pub struct PnPSolver {
    min_inliers: usize,
    max_iterations: usize,
    threshold: f64,
    confidence: f64,
}

impl PnPSolver {
    pub fn new(min_inliers: usize, max_iterations: usize, threshold: f64, confidence: f64) -> Self {
        Self {
            min_inliers,
            max_iterations,
            threshold,
            confidence,
        }
    }

    /// Main PnP solving function with hybrid P3P-EPnP approach and robust outlier rejection
    pub fn solve_pnp(
        &self,
        points_3d: &[na::Point3<f64>],
        points_2d: &[na::Point2<f64>],
        initial_pose: Option<&SE3>,
    ) -> TrackerResult<(SE3, Vec<bool>)> {
        println!("Starting PnP Solver");
        
        if points_3d.len() < MIN_POINTS {
            return Err(TrackerError::InsufficientMatches(
                points_3d.len(),
                MIN_POINTS,
            ));
        }

        // Check for degenerate configurations
        println!("Checking for degenerate configuration");
        if self.is_degenerate_configuration(points_3d) {
            return Err(TrackerError::PoseEstimationFailed(
                "Degenerate point configuration".to_string(),
            ));
        }

        // Normalize points
        println!("Normalizing 3D and 2D points");
        let (normalized_3d, t3d, scale) = self.normalize_points_3d(points_3d);
        let (normalized_2d, t2d) = self.normalize_points_2d(points_2d);

        // Run RANSAC
        println!("Running RANSAC");
        let (best_pose, best_inliers) = self.ransac_pnp(
            &normalized_3d,
            &normalized_2d,
            initial_pose,
            self.threshold / scale,
        )?;

        // Denormalize pose
        println!("Denormalizing pose");
        let pose = self.denormalize_pose(&best_pose, &t3d, &t2d, scale);

        // Final refinement
        println!("Refining pose with Levenberg-Marquardt");
        let final_pose = if best_inliers.iter().filter(|&&x| x).count() >= 6 {
            match self.refine_pose_lm(points_3d, points_2d, &best_inliers, &pose) {
                Ok(refined) => refined,
                Err(_) => pose,
            }
        } else {
            pose
        };

        println!("PnP Solver completed successfully");
        Ok((final_pose, best_inliers))
    }

    fn ransac_pnp(
        &self,
        points_3d: &[na::Point3<f64>],
        points_2d: &[na::Point2<f64>],
        initial_pose: Option<&SE3>,
        threshold: f64,
    ) -> TrackerResult<(SE3, Vec<bool>)> {
        println!("Starting RANSAC with {} points", points_3d.len());
        
        let n_points = points_3d.len();
        let mut rng = rand::thread_rng();
        let mut best_inlier_count = 0;
        let mut best_pose = SE3 {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::zeros(),
        };
        let mut best_inliers = vec![false; n_points];
        let mut iterations = 0;

        // Adaptive RANSAC
        let mut max_iterations = self.max_iterations;

        while iterations < max_iterations {
            println!("RANSAC Iteration {}", iterations);
            let indices: Vec<usize> = (0..n_points).choose_multiple(&mut rng, P3P_POINTS);
            println!("Iteration {}: Selected indices {:?}", iterations, indices);

            if let Some(poses) = self.solve_p3p_grunert(
                &indices.iter().map(|&i| points_3d[i]).collect::<Vec<_>>(),
                &indices.iter().map(|&i| points_2d[i]).collect::<Vec<_>>(),
            ) {
                println!("Found {} candidate poses", poses.len());
                // Evaluate all candidate poses
                for pose in poses {
                    let (inliers, num_inliers) =
                        self.count_inliers(&pose, points_3d, points_2d, threshold);

                    if num_inliers > best_inlier_count {
                        best_inlier_count = num_inliers;
                        best_pose = pose;
                        best_inliers = inliers;

                        // Update number of iterations (adaptive)
                        let inlier_ratio = num_inliers as f64 / n_points as f64;
                        let non_inlier_prob = 1.0 - inlier_ratio.powi(P3P_POINTS as i32);
                        if non_inlier_prob > 0.0 {
                            let new_iterations = (self.confidence.ln()
                                / (1.0 - non_inlier_prob).ln())
                            .ceil() as usize;
                            max_iterations = new_iterations.min(self.max_iterations);
                        }
                    }
                }
            } else {
                println!("P3P solver failed for these points");
            }
            iterations += 1;
        }

        println!("RANSAC finished. Best inlier count: {}", best_inlier_count);

        if best_inlier_count >= self.min_inliers {
            Ok((best_pose, best_inliers))
        } else {
            Err(TrackerError::InsufficientMatches(
                best_inlier_count,
                self.min_inliers,
            ))
        }
    }

    fn solve_p3p_grunert(
        &self,
        points_3d: &[na::Point3<f64>],
        points_2d: &[na::Point2<f64>],
    ) -> Option<Vec<SE3>> {
        println!("P3P input - 3D points: {:?}", points_3d);
        println!("P3P input - 2D points: {:?}", points_2d);

        // Ensure all points are in front of the camera
        if points_3d.iter().any(|p| p.z <= 0.0) {
            println!("Warning: Points with negative or zero Z detected");
            return None;
        }

        // Convert image points to bearing vectors with normalization
        let rays: Vec<na::Unit<na::Vector3<f64>>> = points_2d
            .iter()
            .map(|p| {
                let v = na::Vector3::new(p.x, p.y, 1.0);
                na::Unit::new_normalize(v)
            })
            .collect();

        // Compute distances with better numerical stability
        let dist_12 = (points_3d[1] - points_3d[0]).norm();
        let dist_23 = (points_3d[2] - points_3d[1]).norm();
        let dist_31 = (points_3d[0] - points_3d[2]).norm();

        println!("Distances: d12={}, d23={}, d31={}", dist_12, dist_23, dist_31);

        // Compute angles between rays
        let cos_12 = rays[0].dot(&rays[1]);
        let cos_23 = rays[1].dot(&rays[2]);
        let cos_31 = rays[2].dot(&rays[0]);

        println!("Angles: cos_12={}, cos_23={}, cos_31={}", cos_12, cos_23, cos_31);

        // Validate angles
        if cos_12.abs() >= 1.0 || cos_23.abs() >= 1.0 || cos_31.abs() >= 1.0 {
            println!("Warning: Invalid angles detected");
            return None;
        }

        // Get distance solutions
        let solutions = self.solve_cubic_grunert(
            dist_12, dist_23, dist_31,
            cos_12, cos_23, cos_31,
        )?;

        // Convert distance solutions to poses with validation
        let mut poses = Vec::new();
        for distances in solutions {
            if let Some(pose) = self.distances_to_pose(distances, points_3d, &rays) {
                // Validate pose
                if pose.translation.z > 0.0 {
                    poses.push(pose);
                }
            }
        }

        if poses.is_empty() {
            None
        } else {
            println!("Found {} valid poses", poses.len());
            Some(poses)
        }
    }

    fn solve_cubic_grunert(
        &self,
        d12: f64,
        d23: f64,
        d31: f64,
        cos_12: f64,
        cos_23: f64,
        cos_31: f64,
    ) -> Option<Vec<[f64; 3]>> {
        println!("Input distances: d12={}, d23={}, d31={}", d12, d23, d31);
        println!("Input angles: cos_12={}, cos_23={}, cos_31={}", cos_12, cos_23, cos_31);

        // Normalize distances to improve numerical stability
        let max_dist = d12.max(d23).max(d31);
        let d12_norm = d12 / max_dist;
        let d23_norm = d23 / max_dist;
        let d31_norm = d31 / max_dist;

        let a = d12_norm * d12_norm;
        let b = d23_norm * d23_norm;
        let c = d31_norm * d31_norm;

        // Compute intermediate values with better numerical stability
        let cos_12_sq = cos_12 * cos_12;
        let k1 = (a - b + c) / (2.0 * a);
        let k2 = cos_12;
        let k3 = (a + b - c) / (2.0 * b);
        let k4 = cos_23;

        println!("Intermediate values: k1={}, k2={}, k3={}, k4={}", k1, k2, k3, k4);

        // Modified coefficients for quartic equation
        let a4 = (k1 * k1 + k2 * k2 - 1.0) * (k3 * k3 + k4 * k4 - 1.0);
        let a3 = 4.0 * k1 * k3 * (k1 * k3 + k2 * k4 - 1.0);
        let a2 = 2.0 * ((k1 * k1 + k2 * k2 - 1.0) + (k3 * k3 + k4 * k4 - 1.0));
        let a1 = a3;
        let a0 = a4;

        println!("Quartic coefficients: a4={}, a3={}, a2={}, a1={}, a0={}", 
                 a4, a3, a2, a1, a0);

        // Check if coefficients are valid
        if a4.abs() < 1e-10 {
            println!("Warning: Leading coefficient a4 is near zero");
            return None;
        }

        // Solve quartic equation using companion matrix with improved numerical stability
        let companion = na::Matrix4::new(
            -a3 / a4, -a2 / a4, -a1 / a4, -a0 / a4,
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        );

        let eigenvalues = companion.complex_eigenvalues();
        println!("Eigenvalues: {:?}", eigenvalues);

        // Extract real solutions with improved filtering
        let mut solutions = Vec::new();
        for ev in eigenvalues.iter() {
            if ev.im.abs() < 1e-8 && ev.re > 1e-8 {
                let x = ev.re;
                
                // Prevent division by zero with a threshold
                if b * x < 1e-10 {
                    continue;
                }

                let v = (a - c * x * x) / (2.0 * b * x);
                let u = k1 + k2 * v;

                // Additional validity checks
                if u > 1e-8 && v.is_finite() {
                    // Denormalize distances
                    solutions.push([u * d12, v * d23, x * d31]);
                }
            }
        }

        println!("Number of solutions found: {}", solutions.len());
        if !solutions.is_empty() {
            Some(solutions)
        } else {
            None
        }
    }

    fn distances_to_pose(
        &self,
        distances: [f64; 3],
        points_3d: &[na::Point3<f64>],
        rays: &[na::Unit<na::Vector3<f64>>],
    ) -> Option<SE3> {
        // Compute camera frame points
        let pc1 = rays[0].into_inner() * distances[0];
        let pc2 = rays[1].into_inner() * distances[1];
        let pc3 = rays[2].into_inner() * distances[2];

        let camera_points = [
            na::Point3::from(pc1),
            na::Point3::from(pc2),
            na::Point3::from(pc3),
        ];

        // Solve absolute orientation (Kabsch algorithm)
        let mut centroid_camera = na::Point3::origin();
        let mut centroid_world = na::Point3::origin();

        for i in 0..3 {
            centroid_camera += camera_points[i].coords;
            centroid_world += points_3d[i].coords;
        }
        centroid_camera.coords /= 3.0;
        centroid_world.coords /= 3.0;

        let mut h = na::Matrix3::zeros();
        for i in 0..3 {
            let pw = points_3d[i] - centroid_world;
            let pc = camera_points[i] - centroid_camera;
            h += pc * pw.transpose();
        }

        let svd = h.svd(true, true);
        let (u, v_t) = match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => (u, vt),
            _ => return None,
        };

        // Ensure proper rotation
        let mut r = u * v_t;
        if r.determinant() < 0.0 {
            let mut v_t = v_t;
            v_t.column_mut(2).scale_mut(-1.0);
            r = u * v_t;
        }

        let t = centroid_camera.coords - r * centroid_world.coords;

        Some(SE3 {
            rotation: r,
            translation: t,
        })
    }

    fn refine_pose_lm(
        &self,
        points_3d: &[na::Point3<f64>],
        points_2d: &[na::Point2<f64>],
        inliers: &[bool],
        initial_pose: &SE3,
    ) -> TrackerResult<SE3> {
        let mut pose = initial_pose.clone();
        let mut lambda = INIT_LAMBDA;
        let mut last_error = f64::MAX;

        for _ in 0..LM_MAX_ITERS {
            let (jacobian, residuals, error) =
                self.compute_pose_jacobian(&pose, points_3d, points_2d, inliers);

            if error < last_error {
                lambda /= LAMBDA_FACTOR;
            } else {
                lambda *= LAMBDA_FACTOR;
                continue;
            }

            // Solve normal equations with Levenberg-Marquardt damping
            let mut normal_matrix = jacobian.transpose() * &jacobian;
            for i in 0..6 {
                normal_matrix[(i, i)] += lambda;
            }

            match normal_matrix.try_inverse() {
                Some(inv) => {
                    let update = -inv * jacobian.transpose() * residuals;
                    pose = self.update_pose(&pose, &update);

                    if update.norm() < LM_EPSILON {
                        break;
                    }
                    last_error = error;
                }
                None => break,
            }
        }

        Ok(pose)
    }

    fn is_degenerate_configuration(&self, points: &[na::Point3<f64>]) -> bool {
        if points.len() < 4 {
            println!("Degenerate: Less than 4 points");
            return true;
        }

        // Compute vectors from first point
        let v1 = points[1] - points[0];
        let v2 = points[2] - points[0];
        let normal = v1.cross(&v2);

        println!("Normal Vector: {:?}", normal);

        // Check if the normal vector is significant
        let normal_norm = normal.norm();
        if normal_norm < 1e-6 {
            println!("Degenerate: Normal vector is too small (norm = {})", normal_norm);
            return true;
        }

        // Normalize the normal vector for consistent dot product calculations
        let normal_unit = normal / normal_norm;

        // Check if remaining points lie on the same plane
        for (i, point) in points.iter().skip(3).enumerate() {
            let v = point - points[0];
            let dot = v.dot(&normal_unit);
            println!(
                "Point {}: {:?}, Dot with normal: {}",
                i + 3,
                point,
                dot
            );
            if dot.abs() > 1e-3 {
                println!("Point {} is non-coplanar. Configuration is non-degenerate.", i + 3);
                return false; // Found a non-coplanar point
            }
        }

        println!("All points are coplanar. Configuration is degenerate.");
        true // All points are coplanar
    }

    fn normalize_points_3d(
        &self,
        points: &[na::Point3<f64>],
    ) -> (Vec<na::Point3<f64>>, na::Matrix4<f64>, f64) {
        let mut centroid = na::Point3::origin();
        for point in points {
            centroid += point.coords;
        }
        centroid.coords /= points.len() as f64;

        let mut scale = 0.0;
        for point in points {
            scale += (point - centroid).norm();
        }
        scale /= points.len() as f64;
        let norm_factor = (2.0f64).sqrt() / scale;

        let mut transform = na::Matrix4::identity();
        transform
            .fixed_slice_mut::<3, 1>(0, 3)
            .copy_from(&(-centroid.coords * norm_factor));
        transform
            .fixed_slice_mut::<3, 3>(0, 0)
            .scale_mut(norm_factor);

        let normalized: Vec<_> = points
            .iter()
            .map(|p| {
                na::Point3::from(
                    (transform.fixed_slice::<3, 3>(0, 0) * p.coords)
                        + transform.fixed_slice::<3, 1>(0, 3),
                )
            })
            .collect();

        (normalized, transform, scale)
    }

    fn normalize_points_2d(
        &self,
        points: &[na::Point2<f64>],
    ) -> (Vec<na::Point2<f64>>, na::Matrix3<f64>) {
        let mut centroid = na::Point2::origin();
        for point in points {
            centroid += point.coords;
        }
        centroid.coords /= points.len() as f64;

        let mut scale = 0.0;
        for point in points {
            scale += (point - centroid).norm();
        }
        scale /= points.len() as f64;
        let norm_factor = (2.0f64).sqrt() / scale;

        let mut transform = na::Matrix3::identity();
        transform
            .fixed_slice_mut::<2, 1>(0, 2)
            .copy_from(&(-centroid.coords * norm_factor));
        transform
            .fixed_slice_mut::<2, 2>(0, 0)
            .scale_mut(norm_factor);

        let normalized: Vec<_> = points
            .iter()
            .map(|p| {
                na::Point2::from(
                    (transform.fixed_slice::<2, 2>(0, 0) * p.coords)
                        + transform.fixed_slice::<2, 1>(0, 2),
                )
            })
            .collect();

        (normalized, transform)
    }

    fn denormalize_pose(
        &self,
        pose: &SE3,
        t3d: &na::Matrix4<f64>,
        t2d: &na::Matrix3<f64>,
        scale: f64,
    ) -> SE3 {
        let t2d_inv = t2d.try_inverse().unwrap_or(na::Matrix3::identity());
        let _t3d_inv = t3d.try_inverse().unwrap_or(na::Matrix4::identity());

        let r = t2d_inv.fixed_slice::<3, 3>(0, 0) * pose.rotation * t3d.fixed_slice::<3, 3>(0, 0);

        let t = t2d_inv.fixed_slice::<3, 3>(0, 0)
            * (pose.translation + pose.rotation * t3d.fixed_slice::<3, 1>(0, 3)
                - t2d.fixed_slice::<3, 1>(0, 2) / scale);

        SE3 {
            rotation: r,
            translation: t,
        }
    }

    fn compute_pose_jacobian(
        &self,
        pose: &SE3,
        points_3d: &[na::Point3<f64>],
        points_2d: &[na::Point2<f64>],
        inliers: &[bool],
    ) -> (na::DMatrix<f64>, na::DVector<f64>, f64) {
        let n_inliers = inliers.iter().filter(|&&x| x).count();
        let mut jacobian = na::DMatrix::zeros(n_inliers * 2, 6);
        let mut residuals = na::DVector::zeros(n_inliers * 2);
        let mut total_error = 0.0;
        let mut row = 0;

        for (i, (&point_3d, &point_2d)) in points_3d.iter().zip(points_2d.iter()).enumerate() {
            if !inliers[i] {
                continue;
            }

            // Transform point to camera frame
            let p_cam = pose.rotation * point_3d + pose.translation;

            if p_cam.z <= MIN_DEPTH {
                continue;
            }

            let z_inv = 1.0 / p_cam.z;
            let z_inv_2 = z_inv * z_inv;

            // Compute residuals
            let proj_x = p_cam.x * z_inv;
            let proj_y = p_cam.y * z_inv;
            residuals[2 * row] = proj_x - point_2d.x;
            residuals[2 * row + 1] = proj_y - point_2d.y;

            // Compute Jacobian for rotation (using Rodriguez formula)
            let x = p_cam.x;
            let y = p_cam.y;
            let _z = p_cam.z;

            // For rotation
            jacobian[(2 * row, 0)] = z_inv;
            jacobian[(2 * row, 1)] = 0.0;
            jacobian[(2 * row, 2)] = -x * z_inv_2;

            jacobian[(2 * row + 1, 0)] = 0.0;
            jacobian[(2 * row + 1, 1)] = z_inv;
            jacobian[(2 * row + 1, 2)] = -y * z_inv_2;

            // For translation
            let xy = x * y * z_inv_2;
            let x2 = x * x * z_inv_2;
            let y2 = y * y * z_inv_2;

            jacobian[(2 * row, 3)] = -(1.0 + x2);
            jacobian[(2 * row, 4)] = xy;
            jacobian[(2 * row, 5)] = y * z_inv;

            jacobian[(2 * row + 1, 3)] = xy;
            jacobian[(2 * row + 1, 4)] = -(1.0 + y2);
            jacobian[(2 * row + 1, 5)] = -x * z_inv;

            total_error += residuals[2 * row].powi(2) + residuals[2 * row + 1].powi(2);
            row += 1;
        }

        (jacobian, residuals, total_error)
    }

    fn update_pose(&self, pose: &SE3, update: &na::DVector<f64>) -> SE3 {
        // Extract rotation and translation updates
        let w = na::Vector3::new(update[0], update[1], update[2]);
        let t = na::Vector3::new(update[3], update[4], update[5]);

        // Update rotation using Rodriguez formula
        let theta = w.norm();
        let rotation = if theta < 1e-8 {
            pose.rotation.clone()
        } else {
            let omega_hat =
                na::Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0) / theta;

            let rot_update = na::Matrix3::identity()
                + omega_hat * theta.sin()
                + omega_hat * omega_hat * (1.0 - theta.cos());

            rot_update * pose.rotation
        };

        let translation = pose.translation + t;

        SE3 {
            rotation,
            translation,
        }
    }

    fn count_inliers(
        &self,
        pose: &SE3,
        points_3d: &[na::Point3<f64>],
        points_2d: &[na::Point2<f64>],
        threshold: f64,
    ) -> (Vec<bool>, usize) {
        let mut inliers = vec![false; points_3d.len()];
        let mut num_inliers = 0;

        for (i, (&p3d, &p2d)) in points_3d.iter().zip(points_2d.iter()).enumerate() {
            let p_cam = pose.rotation * p3d + pose.translation;

            if p_cam.z <= MIN_DEPTH || p_cam.z >= MAX_DEPTH {
                continue;
            }

            let proj = na::Point2::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z);

            let error = (proj - p2d).norm();
            if error < threshold {
                inliers[i] = true;
                num_inliers += 1;
            }
        }

        (inliers, num_inliers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pnp_solver_simple_case() {
        let solver = PnPSolver::new(4, 100, 0.01, 0.999);

        // Create well-conditioned non-coplanar test points
        let points_3d = vec![
            na::Point3::new(0.0, 0.0, 4.0),      // Origin point
            na::Point3::new(1.0, 0.0, 4.0),      // X-axis point
            na::Point3::new(0.0, 1.0, 4.0),      // Y-axis point
            na::Point3::new(0.5, 0.5, 5.0),      // Off-plane point
            na::Point3::new(-0.5, -0.5, 4.5),    // Additional point
        ];

        // Generate corresponding 2D points with proper projection
        let points_2d: Vec<na::Point2<f64>> = points_3d.iter()
            .map(|p| {
                let scale = 1.0 / p.z;
                na::Point2::new(p.x * scale, p.y * scale)
            })
            .collect();

        println!("3D Points: {:?}", points_3d);
        println!("2D Points: {:?}", points_2d);

        // Test without initial pose
        let result = solver.solve_pnp(&points_3d, &points_2d, None);
        match &result {
            Ok((pose, inliers)) => {
                println!("Success! Pose: R={:?}, t={:?}", pose.rotation, pose.translation);
                println!("Inliers: {:?}", inliers);
                
                // Verify the pose makes sense
                assert!(pose.translation.z > 0.0, "Camera should be in front of points");
                assert!(inliers.iter().filter(|&&x| x).count() >= 4, 
                       "Should have at least 4 inliers");
            }
            Err(e) => {
                println!("Failed with error: {:?}", e);
            }
        }
        assert!(result.is_ok(), "PnP solver failed without initial pose");
    }

    #[test]
    fn test_degenerate_cases() {
        let solver = PnPSolver::new(4, 100, 0.01, 0.999);

        // Test with coplanar points
        let coplanar_points_3d = vec![
            na::Point3::new(0.0, 0.0, 4.0),
            na::Point3::new(1.0, 0.0, 4.0),
            na::Point3::new(0.0, 1.0, 4.0),
            na::Point3::new(1.0, 1.0, 4.0),
        ];

        let points_2d = vec![
            na::Point2::new(0.0, 0.0),
            na::Point2::new(0.25, 0.0),
            na::Point2::new(0.0, 0.25),
            na::Point2::new(0.25, 0.25),
        ];

        assert!(solver.is_degenerate_configuration(&coplanar_points_3d),
            "Coplanar points should be detected as degenerate");

        // Test with insufficient points
        let insufficient_points_3d = vec![
            na::Point3::new(0.0, 0.0, 4.0),
            na::Point3::new(1.0, 0.0, 4.0),
            na::Point3::new(0.0, 1.0, 4.0),
        ];

        let result = solver.solve_pnp(&insufficient_points_3d, &points_2d[..3], None);
        assert!(matches!(result, Err(TrackerError::InsufficientMatches(_, _))),
            "Should fail with insufficient points");
    }
}