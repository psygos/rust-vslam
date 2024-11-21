use crate::feature_matcher::FeatureMatch;
use crate::pnp::PnPSolver;
use crate::tracker_types::*;
use na::Point2;
use nalgebra as na;
use rand::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
}

impl CameraIntrinsics {
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: u32,
        height: u32,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            k1,
            k2,
            p1,
            p2,
        }
    }

    pub fn unproject(&self, point: &Point2<f64>) -> Point2<f64> {
        let mut x = (point.x - self.cx) / self.fx;
        let mut y = (point.y - self.cy) / self.fy;

        if self.k1 != 0.0 || self.k2 != 0.0 || self.p1 != 0.0 || self.p2 != 0.0 {
            for _ in 0..5 {
                let r2 = x * x + y * y;
                let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
                let tangential_x = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
                let tangential_y = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;

                x = (point.x - self.cx) / self.fx - tangential_x;
                y = (point.y - self.cy) / self.fy - tangential_y;

                x /= radial;
                y /= radial;
            }
        }

        Point2::new(x, y)
    }

    pub fn project(&self, point_3d: &na::Point3<f64>) -> na::Point2<f64> {
        // Early exit for points at or behind camera
        if point_3d.z <= 1e-6 {
            return na::Point2::new(0.0, 0.0);
        }

        // Perspective projection
        let x = point_3d.x / point_3d.z;
        let y = point_3d.y / point_3d.z;

        // Apply distortion if any distortion coefficients are non-zero
        let mut x_distorted = x;
        let mut y_distorted = y;

        if self.k1 != 0.0 || self.k2 != 0.0 || self.p1 != 0.0 || self.p2 != 0.0 {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;

            // Radial distortion
            let radial = 1.0 + self.k1 * r2 + self.k2 * r4;
            x_distorted = x * radial;
            y_distorted = y * radial;

            // Tangential distortion
            let xy_2 = 2.0 * x * y;
            x_distorted += self.p1 * xy_2 + self.p2 * (r2 + 2.0 * x * x);
            y_distorted += self.p1 * (r2 + 2.0 * y * y) + self.p2 * xy_2;
        }

        // Apply camera matrix
        na::Point2::new(
            self.fx * x_distorted + self.cx,
            self.fy * y_distorted + self.cy,
        )
    }
}

pub struct PoseEstimationResult {
    pub pose: SE3,
    pub inliers: Vec<bool>,
    pub points_3d: Vec<na::Point3<f64>>,
    pub triangulated_indices: Vec<usize>,
    pub score: f64,
}

pub struct PoseEstimator {
    camera: Arc<CameraIntrinsics>,
    params: RansacParams,
    pnp_solver: PnPSolver,
}

impl PoseEstimator {
    pub fn new(camera: Arc<CameraIntrinsics>, params: RansacParams) -> Self {
        Self {
            camera,
            params,
            pnp_solver: PnPSolver::new(
                params.min_inliers,
                params.max_iterations,
                params.threshold,
                params.confidence,
            ),
        }
    }

    pub fn estimate_pose(
        &self,
        matches: &[FeatureMatch],
        frame1: &Frame,
        frame2: &Frame,
        initial_guess: Option<&SE3>,
    ) -> TrackerResult<PoseEstimationResult> {
        if matches.len() < 8 {
            return Err(TrackerError::InsufficientMatches(matches.len(), 8));
        }

        let mut points1 = Vec::with_capacity(matches.len());
        let mut points2 = Vec::with_capacity(matches.len());

        for m in matches {
            let kp1 = &frame1.features[m.query_idx];
            let kp2 = &frame2.features[m.train_idx];
            points1.push(
                self.camera
                    .unproject(&na::Point2::new(kp1.point.x, kp1.point.y)),
            );
            points2.push(
                self.camera
                    .unproject(&na::Point2::new(kp2.point.x, kp2.point.y)),
            );
        }

        let (essential_matrix, inliers) =
            self.estimate_essential_matrix(&points1, &points2, matches, initial_guess)?;

        let (points_3d, triangulated_indices) =
            self.triangulate_points(&essential_matrix, &points1, &points2, &inliers)?;

        let (r, t) = self.decompose_essential_matrix(&essential_matrix)?;
        let _depth_score = self.verify_reconstruction(&points_3d)?;

        let pose = SE3 {
            rotation: r,
            translation: t,
        };

        let score = self.compute_pose_score(&points_3d, &inliers);

        Ok(PoseEstimationResult {
            pose,
            inliers,
            points_3d,
            triangulated_indices,
            score,
        })
    }

    fn estimate_essential_matrix(
        &self,
        points1: &[na::Point2<f64>],
        points2: &[na::Point2<f64>],
        matches: &[FeatureMatch],
        _initial_guess: Option<&SE3>,
    ) -> TrackerResult<(na::Matrix3<f64>, Vec<bool>)> {
        if points1.len() != points2.len() || points1.len() < 8 {
            return Err(TrackerError::InsufficientMatches(points1.len(), 8));
        }

        let mut best_matrix = na::Matrix3::zeros();
        let mut best_inliers = vec![false; matches.len()];
        let mut best_inlier_count = 0;
        let mut rng = thread_rng();

        let indices: Vec<usize> = (0..points1.len()).collect();

        for _ in 0..self.params.max_iterations {
            let sample: Vec<_> = indices.choose_multiple(&mut rng, 8).copied().collect();

            if let Some(e_mat) = self.compute_essential_matrix_minimal(&sample, points1, points2) {
                let inliers = self.find_inliers(&e_mat, points1, points2, self.params.threshold);
                let inlier_count = inliers.iter().filter(|&&x| x).count();

                if inlier_count > best_inlier_count {
                    best_matrix = e_mat;
                    best_inliers = inliers;
                    best_inlier_count = inlier_count;
                }
            }
        }

        if best_inlier_count >= self.params.min_inliers {
            Ok((best_matrix, best_inliers))
        } else {
            Err(TrackerError::InsufficientMatches(
                best_inlier_count,
                self.params.min_inliers,
            ))
        }
    }

    fn compute_essential_matrix_minimal(
        &self,
        indices: &[usize],
        points1: &[na::Point2<f64>],
        points2: &[na::Point2<f64>],
    ) -> Option<na::Matrix3<f64>> {
        if indices.len() != 8 {
            return None;
        }

        let mut a = na::DMatrix::zeros(8, 9); // Adjusted size for 8 points

        for (i, &idx) in indices.iter().enumerate() {
            let p1 = points1.get(idx)?; // Use get() with ? for bounds checking
            let p2 = points2.get(idx)?;

            let row = [
                p1.x * p2.x,
                p1.y * p2.x,
                p2.x,
                p1.x * p2.y,
                p1.y * p2.y,
                p2.y,
                p1.x,
                p1.y,
                1.0,
            ];

            a.row_mut(i).copy_from_slice(&row); // Use copy_from_slice
        }

        let svd = a.svd(true, true);
        let (u, v_t) = match (svd.u, svd.v_t) {
            (Some(u), Some(v_t)) => (u, v_t),
            _ => return None,
        };

        let v = v_t.transpose();
        let nullspace = v.column(v.ncols() - 1).into_owned(); // Get the last column

        let e = na::Matrix3::from_column_slice((&nullspace).into()); // Use from_column_slice

        // Enforce essential matrix properties (correcting potential issues here)
        let svd_e = e.svd(true, true);
        let (u_e, v_t_e) = match (svd_e.u, svd_e.v_t) {
            (Some(u), Some(vt)) => (u, vt),
            _ => return None,
        };
        let s = na::Matrix3::from_diagonal(&na::Vector3::new(1.0, 1.0, 0.0)); //Corrected singular value matrix
        Some(u_e * s * v_t_e)
    }

    fn find_inliers(
        &self,
        e_mat: &na::Matrix3<f64>,
        points1: &[na::Point2<f64>],
        points2: &[na::Point2<f64>],
        threshold: f64,
    ) -> Vec<bool> {
        points1
            .iter()
            .zip(points2.iter())
            .map(|(p1, p2)| {
                let p1_h = na::Vector3::new(p1.x, p1.y, 1.0);
                let p2_h = na::Vector3::new(p2.x, p2.y, 1.0);
                let epiline = e_mat * p1_h;

                if epiline.norm() < 1e-8 {
                    return false;
                }

                let error = (p2_h.transpose() * e_mat * p1_h).abs()[0] / epiline.norm();
                error < threshold
            })
            .collect()
    }

    fn decompose_essential_matrix(
        &self,
        e_mat: &na::Matrix3<f64>,
    ) -> TrackerResult<(na::Matrix3<f64>, na::Vector3<f64>)> {
        let svd = e_mat.svd(true, true);
        let (u, vt) = match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => (u, vt),
            _ => return Err(TrackerError::PoseEstimationFailed("SVD failed".to_string())),
        };

        let w = na::Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let mut r = u * w * vt;
        let mut t = u.column(2).into_owned();

        // Ensure proper rotation matrix
        if r.determinant() < 0.0 {
            r *= -1.0;
            t *= -1.0;
        }

        // Normalize translation
        let t_norm = t.norm();
        if t_norm > 1e-8 {
            t /= t_norm;
        }

        Ok((r, t))
    }

    fn triangulate_points(
        &self,
        e_mat: &na::Matrix3<f64>,
        points1: &[na::Point2<f64>],
        points2: &[na::Point2<f64>],
        inliers: &[bool],
    ) -> TrackerResult<(Vec<na::Point3<f64>>, Vec<usize>)> {
        let (r, t) = self.decompose_essential_matrix(e_mat)?;
        let mut points_3d = Vec::new();
        let mut triangulated_indices = Vec::new();

        for (idx, ((&p1, &p2), &is_inlier)) in points1
            .iter()
            .zip(points2.iter())
            .zip(inliers.iter())
            .enumerate()
        {
            if !is_inlier {
                continue;
            }

            if let Some(point) = self.triangulate_point(&p1, &p2, &r, &t)? {
                points_3d.push(point);
                triangulated_indices.push(idx);
            }
        }

        Ok((points_3d, triangulated_indices))
    }

    fn triangulate_point(
        &self,
        p1: &na::Point2<f64>,
        p2: &na::Point2<f64>,
        r: &na::Matrix3<f64>,
        t: &na::Vector3<f64>,
    ) -> TrackerResult<Option<na::Point3<f64>>> {
        let mut a = na::Matrix4::zeros();

        // First camera matrix [I|0]
        let p1_cross = na::Matrix3::new(0.0, -1.0, p1.y, 1.0, 0.0, -p1.x, -p1.y, p1.x, 0.0);

        // Second camera matrix [R|t]
        let p2_cross = na::Matrix3::new(0.0, -1.0, p2.y, 1.0, 0.0, -p2.x, -p2.y, p2.x, 0.0);

        // Construct full projection matrices
        let p1_mat = p1_cross * na::Matrix3x4::identity();
        let p2_mat = p2_cross
            * na::Matrix3x4::from_columns(&[
                r.column(0).into_owned(),
                r.column(1).into_owned(),
                r.column(2).into_owned(),
                t.clone(),
            ]);

        // Fill DLT matrix (corrected)
        a.fixed_slice_mut::<2, 4>(0, 0)
            .copy_from(&p1_mat.fixed_slice::<2, 4>(0, 0));
        a.fixed_slice_mut::<2, 4>(2, 0)
            .copy_from(&p2_mat.fixed_slice::<2, 4>(0, 0));

        let svd = a.svd(true, true);
        let v_t = svd.v_t;
        if v_t.is_none() {
            return Err(TrackerError::PoseEstimationFailed(
                "SVD failed for triangulation".to_string(),
            ));
        }
        let v = v_t.unwrap().transpose();

        let point_h = v.column(3);
        if point_h[3].abs() < 1e-8 {
            return Ok(None);
        }

        let point = na::Point3::new(
            point_h[0] / point_h[3],
            point_h[1] / point_h[3],
            point_h[2] / point_h[3],
        );

        let t_vec = t.clone(); // Use a Vector3 for translation
        let transformed = r * point + t_vec;

        if point.z > 0.0 && transformed.z > 0.0 {
            Ok(Some(point))
        } else {
            Ok(None)
        }
    }

    pub fn estimate_pose_pnp(
        &self,
        points_3d: &[na::Point3<f64>],
        points_2d: &[na::Point2<f64>],
        initial_pose: Option<&SE3>,
    ) -> TrackerResult<PoseEstimationResult> {
        // First normalize points using camera intrinsics
        let normalized_points: Vec<na::Point2<f64>> =
            points_2d.iter().map(|p| self.camera.unproject(p)).collect();

        // Solve PnP
        let (pose, inliers) =
            self.pnp_solver
                .solve_pnp(points_3d, &normalized_points, initial_pose)?;

        let inlier_count = inliers.iter().filter(|&&x| x).count();

        Ok(PoseEstimationResult {
            pose,
            inliers,
            points_3d: points_3d.to_vec(),
            triangulated_indices: (0..points_3d.len()).collect(),
            score: inlier_count as f64 / points_3d.len() as f64,
        })
    }

    fn verify_reconstruction(&self, points: &[na::Point3<f64>]) -> TrackerResult<f64> {
        let mut valid_points = 0;
        let mut total_depth = 0.0;

        for point in points {
            if point.z > 0.0 {
                valid_points += 1;
                total_depth += point.z;
            }
        }

        if valid_points > 0 {
            Ok(total_depth / valid_points as f64)
        } else {
            Err(TrackerError::PoseEstimationFailed(
                "No valid points".to_string(),
            ))
        }
    }

    fn compute_pose_score(&self, points_3d: &[na::Point3<f64>], inliers: &[bool]) -> f64 {
        let inlier_ratio = inliers.iter().filter(|&&x| x).count() as f64 / inliers.len() as f64;
        let point_score =
            points_3d.iter().filter(|p| p.z > 0.0).count() as f64 / points_3d.len() as f64;

        inlier_ratio * point_score
    }
}
