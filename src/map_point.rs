use nalgebra as na;
use std::collections::HashMap;
use std::sync::Arc;

use crate::feature_matcher::FeatureMatch;
use crate::pose_estimator::CameraIntrinsics;
use crate::tracker_types::{Frame, TrackerError, TrackerResult, SE3};

const MAX_REPROJ_ERROR: f64 = 4.0; // Slightly more permissive
const MIN_VIEW_DOT: f64 = 0.2; // More permissive angle ~78 degrees
pub struct MapPoint {
    pub id: usize,
    pub position: na::Point3<f64>,
    pub first_keyframe_id: usize,
    pub first_keyframe_pose: SE3,
    pub observations: HashMap<usize, na::Point2<f64>>,
    pub descriptor: Vec<u8>,
    pub normal: na::Vector3<f64>,
    pub min_distance: f64,
    pub max_distance: f64,
    pub tracking_score: f64,
    pub last_observed_frame_id: usize,
    pub num_visible: usize,
    pub num_found: usize,
}

pub struct LocalMap {
    pub keyframe_poses: HashMap<usize, SE3>,
    pub points: HashMap<usize, MapPoint>,
    camera: Arc<CameraIntrinsics>,
    next_point_id: usize,
    min_observations: usize,
}

impl LocalMap {
    pub fn new(camera: Arc<CameraIntrinsics>) -> Self {
        Self {
            keyframe_poses: HashMap::new(),
            points: HashMap::new(),
            camera,
            next_point_id: 0,
            min_observations: 2,
        }
    }
    pub fn get_point_count(&self) -> usize {
        self.points.len()
    }

    pub fn add_keyframe(&mut self, frame_id: usize, pose: SE3) {
        self.keyframe_poses.insert(frame_id, pose);
    }

    pub fn triangulate_points(
        &mut self,
        kf1_id: usize,
        kf2_id: usize,
        matches: &[FeatureMatch],
        kf1: &Frame,
        kf2: &Frame,
    ) -> TrackerResult<Vec<usize>> {
        let pose1 = self.keyframe_poses.get(&kf1_id).ok_or_else(|| {
            TrackerError::InitializationFailed(format!("Missing keyframe {}", kf1_id))
        })?;

        let pose2 = self.keyframe_poses.get(&kf2_id).ok_or_else(|| {
            TrackerError::InitializationFailed(format!("Missing keyframe {}", kf2_id))
        })?;

        let mut new_point_ids = Vec::new();

        for m in matches {
            // Skip if match indices are invalid
            if m.query_idx >= kf1.features.len() || m.train_idx >= kf2.features.len() {
                continue;
            }

            let p1 = &kf1.features[m.query_idx].point;
            let p2 = &kf2.features[m.train_idx].point;

            // Skip if point already exists
            if self.points.values().any(|p| {
                p.observations.contains_key(&kf1_id) && (p.observations[&kf1_id] - p1).norm() < 1e-4
            }) {
                continue;
            }

            if let Some(point_3d) = self.triangulate_point(p1, p2, pose1, pose2)? {
                let view_dir = (point_3d - pose1.translation).coords.normalize();
                let distance = (point_3d - pose1.translation).coords.norm();

                let mut map_point = MapPoint {
                    id: self.next_point_id,
                    position: point_3d,
                    first_keyframe_id: kf1_id,
                    first_keyframe_pose: pose1.clone(),
                    observations: HashMap::new(),
                    descriptor: kf1.descriptors[m.query_level][m.query_idx]
                        .descriptor
                        .clone(),
                    normal: view_dir,
                    min_distance: distance,
                    max_distance: distance,
                    tracking_score: 1.0,
                    last_observed_frame_id: kf2_id,
                    num_visible: 1,
                    num_found: 1,
                };

                // Add observations from both views
                map_point.observations.insert(kf1_id, p1.clone());
                map_point.observations.insert(kf2_id, p2.clone());

                self.points.insert(self.next_point_id, map_point);
                new_point_ids.push(self.next_point_id);
                self.next_point_id += 1;
            }
        }

        Ok(new_point_ids)
    }

    fn triangulate_point(
        &self,
        p1: &na::Point2<f64>,
        p2: &na::Point2<f64>,
        pose1: &SE3,
        pose2: &SE3,
    ) -> TrackerResult<Option<na::Point3<f64>>> {
        let p1_norm = na::Point2::new(
            (p1.x - self.camera.cx) / self.camera.fx,
            (p1.y - self.camera.cy) / self.camera.fy,
        );

        let p2_norm = na::Point2::new(
            (p2.x - self.camera.cx) / self.camera.fx,
            (p2.y - self.camera.cy) / self.camera.fy,
        );

        // Build projection matrices
        let mut t1 = na::Matrix4::identity();
        let mut t2 = na::Matrix4::identity();

        t1.fixed_slice_mut::<3, 3>(0, 0).copy_from(&pose1.rotation);
        t1.fixed_slice_mut::<3, 1>(0, 3)
            .copy_from(&pose1.translation);

        t2.fixed_slice_mut::<3, 3>(0, 0).copy_from(&pose2.rotation);
        t2.fixed_slice_mut::<3, 1>(0, 3)
            .copy_from(&pose2.translation);

        let mut a = na::Matrix4::zeros();

        let skew1 = na::Matrix3::new(
            0.0, -1.0, p1_norm.y, 1.0, 0.0, -p1_norm.x, -p1_norm.y, p1_norm.x, 0.0,
        );

        let skew2 = na::Matrix3::new(
            0.0, -1.0, p2_norm.y, 1.0, 0.0, -p2_norm.x, -p2_norm.y, p2_norm.x, 0.0,
        );

        a.fixed_slice_mut::<2, 4>(0, 0)
            .copy_from(&(skew1 * t1.fixed_slice::<3, 4>(0, 0)).fixed_slice::<2, 4>(0, 0));
        a.fixed_slice_mut::<2, 4>(2, 0)
            .copy_from(&(skew2 * t2.fixed_slice::<3, 4>(0, 0)).fixed_slice::<2, 4>(0, 0));

        let svd = a.svd(true, true);
        let v = svd
            .v_t
            .ok_or(TrackerError::InitializationFailed("SVD failed".to_string()))?
            .transpose();
        let point_h = v.column(3);

        if point_h[3].abs() < 1e-8 {
            return Ok(None);
        }

        let point = na::Point3::new(
            point_h[0] / point_h[3],
            point_h[1] / point_h[3],
            point_h[2] / point_h[3],
        );

        // Check reprojection error
        let proj1 = self.camera.as_ref().project(&point);
        let proj2 = self
            .camera
            .as_ref()
            .project(&(pose2.rotation * point + pose2.translation));

        let error1 = (proj1 - p1).norm();
        let error2 = (proj2 - p2).norm();

        if error1 > MAX_REPROJ_ERROR || error2 > MAX_REPROJ_ERROR {
            return Ok(None);
        }

        Ok(Some(point))
    }

    pub fn get_visible_points(
        &self,
        pose: &SE3,
        min_observations: Option<usize>,
    ) -> Vec<(&MapPoint, na::Point2<f64>)> {
        let min_obs = min_observations.unwrap_or(self.min_observations);

        self.points
            .values()
            .filter(|p| p.observations.len() >= min_obs)
            .filter_map(|p| {
                let viewing_direction =
                    (p.position - na::Point3::from(pose.translation)).normalize();

                // Viewing angle check
                if viewing_direction.dot(&p.normal) < MIN_VIEW_DOT {
                    return None;
                }

                // Distance check
                let distance = (p.position - na::Point3::from(pose.translation)).norm();
                if distance < p.min_distance * 0.8 || distance > p.max_distance * 1.2 {
                    return None;
                }

                // Project point
                let proj = Some(na::Point2::new(
                    self.camera.fx * viewing_direction.x / viewing_direction.z + self.camera.cx,
                    self.camera.fy * viewing_direction.y / viewing_direction.z + self.camera.cy,
                ));

                // Check if projection is within image bounds
                proj.filter(|proj_point| {
                    proj_point.x >= 0.0
                        && proj_point.x < self.camera.width as f64
                        && proj_point.y >= 0.0
                        && proj_point.y < self.camera.height as f64
                })
                .map(|proj_point| (p, proj_point))
            })
            .collect()
    }

    pub fn cleanup_old_points(&mut self, current_frame_id: usize, max_age: usize) {
        self.points.retain(|_, p| {
            let age = current_frame_id - p.last_observed_frame_id;
            let good_observations = p.observations.len() > 2;
            let good_tracking = p.num_found as f64 / p.num_visible.max(1) as f64 > 0.25;

            age < max_age || (good_observations && good_tracking)
        });
    }

    pub fn clear(&mut self) {
        self.keyframe_poses.clear();
        self.points.clear();
        self.next_point_id = 0;
    }
}
