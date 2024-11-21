use crate::{
    feature_manager::FeatureManager,
    feature_matcher::{FeatureMatch, OrbMatcher},
    map_point::LocalMap,
    motion_refinement::MotionOptimizer,
    new_distributed_extraction::{compute_descriptor_quad_tree, OrbDescriptor},
    pose_estimator::{CameraIntrinsics, PoseEstimationResult, PoseEstimator},
    pyramid_new::ImagePyramid,
    tracker_types::{
        Feature, Frame, RansacParams, TrackedFeature, TrackerError, TrackerResult, SE3,
    },
};
use image::{ImageBuffer, Rgb};
use nalgebra as na;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

const MIN_PARALLAX_DEG: f64 = 2.5;
const MIN_TRACKED_POINTS: usize = 50;
const MIN_MAP_POINTS: usize = 30;

const INIT_FRAMES: usize = 2;
const KEYFRAME_INTERVAL: usize = 3;

const SCALE_WINDOW: usize = 8;

const MAX_FEATURES: usize = 1000;
const MAX_PYRAMID_LEVELS: usize = 4;
const FAST_MIN_FEATURES: u8 = 20;
const FAST_MAX_FEATURES: u8 = 20;

const PNP_MIN_SCORE: f64 = 0.5;
const PNP_REFINEMENT_THRESHOLD: f64 = 0.3;
const PNP_MAX_REPROJ_ERROR: f64 = 2.0;

const MAX_VIEWING_ANGLE: f64 = 30.0;

#[derive(Debug)]
enum TrackingState {
    Initializing,
    Tracking,
    Lost,
}

pub struct VisualOdometry {
    // Core components
    feature_manager: FeatureManager,
    matcher: OrbMatcher,
    pose_estimator: PoseEstimator,
    motion_optimizer: MotionOptimizer,
    local_map: LocalMap,
    camera: Arc<CameraIntrinsics>,

    // State
    tracking_state: TrackingState,
    cumulative_pose: SE3,
    last_frame: Option<Frame>,
    last_keyframe: Option<Frame>,

    // Simplified motion tracking
    last_pose_change: Option<SE3>,
    scale_history: VecDeque<f64>,

    // Statistics
    frames_since_keyframe: usize,
    num_tracked_points: usize,
    tracking_quality: f64,
    pyramid: Option<ImagePyramid>,

    next_keyframe_id: usize,
    current_time: f64,
    frame_rate: f64,
    frame_delta: f64,
}

impl VisualOdometry {
    pub fn new(camera: Arc<CameraIntrinsics>, frame_rate: f64) -> Self {
        Self {
            feature_manager: FeatureManager::new((8, 8), 1000),
            matcher: OrbMatcher::default(),
            pose_estimator: PoseEstimator::new(
                camera.clone(),
                RansacParams {
                    min_inliers: MIN_TRACKED_POINTS,
                    max_iterations: 200,
                    threshold: 0.01,
                    confidence: 0.999,
                },
            ),
            motion_optimizer: MotionOptimizer::new(camera.clone()),
            local_map: LocalMap::new(camera.clone()),
            camera,
            tracking_state: TrackingState::Initializing,
            cumulative_pose: SE3::identity(),
            last_frame: None,
            last_keyframe: None,
            last_pose_change: None,
            scale_history: VecDeque::with_capacity(SCALE_WINDOW),
            frames_since_keyframe: 0,
            num_tracked_points: 0,
            tracking_quality: 1.0,
            pyramid: None,
            next_keyframe_id: 0,
            current_time: 0.0,
            frame_rate,
            frame_delta: 1.0 / frame_rate,
        }
    }

    pub fn process_image(
        &mut self,
        frame_id: usize,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> TrackerResult<SE3> {
        let timer = Instant::now();

        let gray_frame = image::DynamicImage::ImageRgb8(image.clone()).into_luma8();
        let pyramid = ImagePyramid::new(&gray_frame, MAX_PYRAMID_LEVELS, 1.1);
        let descriptors = compute_descriptor_quad_tree(
            &pyramid,
            FAST_MIN_FEATURES,
            FAST_MAX_FEATURES,
            MAX_FEATURES,
        );
        self.pyramid = Some(pyramid);

        let frame = Frame::new(
            frame_id,
            self.current_time,
            convert_descriptors_to_features(&descriptors),
            descriptors, // Store full ORB descriptors, not bytes
            self.camera.clone(),
            None,
        );

        // Predict pose using motion model
        self.predict_pose(frame.timestamp)?;

        // Increment the timestamp for the next frame
        self.current_time += self.frame_delta;

        self.process_frame(frame)
    }

    fn predict_pose(&mut self, _current_timestamp: f64) -> TrackerResult<()> {
        // Simple motion prediction using last pose change
        if let Some(last_change) = &self.last_pose_change {
            self.cumulative_pose = self.cumulative_pose.compose(last_change);
        }
        Ok(())
    }

    fn update_motion_model(&mut self, pose: &SE3) {
        // Simply store the last pose change
        self.last_pose_change = Some(pose.clone());
    }

    fn process_frame(&mut self, frame: Frame) -> TrackerResult<SE3> {
        // Debug tracking state
        println!(
            "Processing frame {} in state {:?}",
            frame.id, self.tracking_state
        );

        let result = match self.tracking_state {
            TrackingState::Initializing => self.initialize_sequence(frame.clone()),
            TrackingState::Tracking => self.track_frame_with_motion_model(frame.clone()),
            TrackingState::Lost => self.recover_tracking(&frame),
        };

        // Log tracking quality
        if let Ok(pose) = &result {
            println!(
                "Frame {}: Tracked points: {}, Quality: {:.2}",
                frame.id, self.num_tracked_points, self.tracking_quality
            );
        }

        if let Ok(pose) = &result {
            // Update motion model with the newly estimated pose
            self.update_motion_model(pose);
        }

        result
    }

    fn track_frame_with_motion_model(&mut self, frame: Frame) -> TrackerResult<SE3> {
        let last_frame = self.last_frame.as_ref().unwrap();
        let pyramid = self.pyramid.as_ref().unwrap();
        // 1. Project map points and predict matches
        let visible_points = self.local_map
            .get_visible_points(&self.cumulative_pose, None)
            .into_iter()
            .filter(|(point, proj)| {
                // Filter based on viewing angle
                let viewing_angle = compute_viewing_angle(&self.cumulative_pose, &point.position);
                viewing_angle < MAX_VIEWING_ANGLE
            })
            .collect::<Vec<_>>();
        println!("Visible map points: {}", visible_points.len());

        // 2. Feature matching with motion prediction
        // Predict_pose is already called before tracking
        let predicted_pose = if let Some(velocity) = &self.last_pose_change {
            self.cumulative_pose.compose(velocity)
        } else {
            self.cumulative_pose.clone()
        };

        // Match both frame-to-frame and with map points
        let matches = self.matcher.match_descriptors(
            &last_frame.descriptors,
            &frame.descriptors,
            &pyramid.scale_factors,
        );

        if matches.len() < MIN_TRACKED_POINTS {
            println!("Warning: Low number of matches: {}", matches.len());
            self.tracking_state = TrackingState::Lost;
            return Err(TrackerError::InsufficientMatches(
                matches.len(),
                MIN_TRACKED_POINTS,
            ));
        }

        // 3. Pose Estimation
        let tracked_features = self.feature_manager.track_features(
            last_frame,
            &frame,
            &matches,
            &frame.descriptors,
        )?;

        // Try PnP if we have enough map points
        let pose_result = if visible_points.len() >= MIN_MAP_POINTS {
            let points_3d: Vec<_> = visible_points.iter().map(|(p, _)| p.position).collect();
            let points_2d: Vec<_> = visible_points
                .iter()
                .map(|(_, proj)| proj.clone())
                .collect();

            match self.pose_estimator.estimate_pose_pnp(
                &points_3d,
                &points_2d,
                Some(&predicted_pose),
            ) {
                Ok(pose) => pose,
                Err(_) => self.estimate_pose_essential(
                    &matches,
                    last_frame,
                    &frame,
                    Some(&predicted_pose),
                )?,
            }
        } else {
            self.estimate_pose_essential(&matches, last_frame, &frame, Some(&predicted_pose))?
        };

        // 4. Refine pose using motion optimizer
        let points_2d: Vec<_> = pose_result
            .triangulated_indices
            .iter()
            .map(|&idx| frame.features[matches[idx].train_idx].point)
            .collect();

        let refined_pose = self.motion_optimizer.refine_motion(
            &pose_result.pose,
            &pose_result.points_3d,
            &points_2d,
            &vec![1.0; points_2d.len()],
        )?;

        // 5. Update motion model and scale
        self.update_motion_model(&refined_pose);
        let scaled_pose = self.apply_scale_consistency(refined_pose);

        // 6. Keyframe decision
        self.frames_since_keyframe += 1;
        if self.should_create_keyframe(&tracked_features) {
            self.create_keyframe(frame.clone(), scaled_pose.clone())?;
        }

        // 7. Update state
        self.num_tracked_points = tracked_features.len();
        self.tracking_quality = tracked_features.len() as f64 / matches.len() as f64;
        self.cumulative_pose = self.cumulative_pose.compose(&scaled_pose);
        self.last_frame = Some(frame);

        Ok(scaled_pose)
    }

    fn initialize_sequence(&mut self, frame: Frame) -> TrackerResult<SE3> {
        // First frame case
        if self.last_frame.is_none() {
            println!("Initializing with first frame {}", frame.id);
            self.last_frame = Some(frame.clone());
            self.last_keyframe = self.last_frame.clone();
            self.local_map.add_keyframe(frame.id, SE3::identity());
            self.next_keyframe_id = frame.id + 1;
            return Ok(SE3::identity());
        }

        let last_frame = self.last_frame.as_ref().unwrap();
        let pyramid = self.pyramid.as_ref().unwrap();
        
        // Match features between frames
        let matches = self.matcher.match_descriptors(
            &last_frame.descriptors,
            &frame.descriptors,
            &pyramid.scale_factors,
        );

        if matches.len() < MIN_TRACKED_POINTS {
            println!("Insufficient matches for initialization: {}", matches.len());
            self.last_frame = Some(frame);
            return Ok(SE3::identity());
        }

        // Estimate relative pose
        let pose_result = self
            .pose_estimator
            .estimate_pose(&matches, last_frame, &frame, None)?;

        // Check parallax
        if !self.check_sufficient_parallax(&pose_result) {
            println!("Insufficient parallax for initialization");
            self.last_frame = Some(frame);
            return Ok(SE3::identity());
        }

        // Initialize mapping with the first two keyframes
        println!("Initializing map with {} matches", matches.len());
        
        let first_kf = self.last_keyframe.as_ref().unwrap();
        let first_kf_id = first_kf.id;
        let second_kf_id = frame.id;

        // Ensure first keyframe is in the map
        if !self.local_map.keyframe_poses.contains_key(&first_kf_id) {
            self.local_map.add_keyframe(first_kf_id, SE3::identity());
        }
        
        // Add second keyframe
        self.local_map.add_keyframe(second_kf_id, pose_result.pose.clone());

        // Triangulate points between the keyframes
        self.local_map.triangulate_points(
            first_kf_id,
            second_kf_id,
            &matches,
            first_kf,
            &frame,
        )?;

        println!("Initial map created with {} points", self.local_map.get_point_count());

        // Update state
        self.tracking_state = TrackingState::Tracking;
        self.last_keyframe = Some(frame.clone());
        self.last_frame = Some(frame);
        self.num_tracked_points = matches.len();
        self.next_keyframe_id = second_kf_id + 1;

        Ok(pose_result.pose)
    }

    fn check_sufficient_parallax(&self, pose_result: &PoseEstimationResult) -> bool {
        // Compute median parallax from triangulated points
        let parallax = pose_result
            .points_3d
            .iter()
            .filter_map(|p| {
                let depth = p.coords.norm();
                if depth > 0.1 && depth < 40.0 {
                    Some(p.coords.normalize())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if parallax.len() < MIN_TRACKED_POINTS / 2 {
            return false;
        }

        // Compute median angle between rays
        let median_angle = compute_median_parallax(&parallax);
        median_angle >= MIN_PARALLAX_DEG
    }

    fn apply_scale_consistency(&mut self, pose: SE3) -> SE3 {
        let translation_norm = pose.translation.norm();

        // Update scale history
        if self.scale_history.len() >= SCALE_WINDOW {
            self.scale_history.pop_front();
        }
        self.scale_history.push_back(translation_norm);

        // Use map points for scale reference when available
        let map_scale = if let Some(kf) = &self.last_keyframe {
            let visible_points = self
                .local_map
                .get_visible_points(&self.cumulative_pose, None);
            if !visible_points.is_empty() {
                let mut depths: Vec<f64> = visible_points
                    .iter()
                    .map(|(p, _)| {
                        (p.position - self.cumulative_pose.translation)
                            .coords
                            .norm()
                    })
                    .collect();
                depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                Some(depths[depths.len() / 2])  // Median depth
            } else {
                None
            }
        } else {
            None
        };

        // Compute median scale from history
        let mut scales = self.scale_history.iter().cloned().collect::<Vec<_>>();
        scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_scale = scales[scales.len() / 2];

        // Apply scale correction
        let mut adjusted_pose = pose;
        if let Some(map_scale) = map_scale {
            // More conservative thresholds
            if translation_norm > median_scale * 1.2 || translation_norm < median_scale * 0.8 {
                let scale_factor = map_scale / translation_norm;
                adjusted_pose.translation *= scale_factor;
                println!("Scale correction (map-based): {:.3}", scale_factor);
            }
        } else if scales.len() >= 3 {
            // More conservative thresholds
            if translation_norm > median_scale * 1.2 || translation_norm < median_scale * 0.8 {
                adjusted_pose.translation *= median_scale / translation_norm;
                println!("Scale correction (history-based): {:.3}", median_scale / translation_norm);
            }
        }

        adjusted_pose
    }

    fn estimate_pose_essential(
        &self,
        matches: &[FeatureMatch],
        last_frame: &Frame,
        current_frame: &Frame,
        initial_pose: Option<&SE3>,
    ) -> TrackerResult<PoseEstimationResult> {
        self.pose_estimator
            .estimate_pose(matches, last_frame, current_frame, initial_pose)
    }

    fn should_create_keyframe(&self, features: &[TrackedFeature]) -> bool {
        // Don't create keyframe if we just started
        if self.frames_since_keyframe < KEYFRAME_INTERVAL {
            return false;
        }

        // Check tracking quality
        let long_tracks = features.iter().filter(|f| f.positions.len() >= 3).count();

        let tracking_ratio = long_tracks as f64 / features.len() as f64;

        // Create keyframe if:
        // 1. Tracking quality is declining
        // 2. Not enough map points are visible
        // 3. Sufficient motion has occurred
        let visible_points = self
            .local_map
            .get_visible_points(&self.cumulative_pose, None);

        if tracking_ratio < 0.5 || visible_points.len() < MIN_MAP_POINTS {
            println!("Keyframe needed: low tracking quality or few visible points");
            return true;
        }

        // Check motion since last keyframe
        if let Some(last_kf) = &self.last_keyframe {
            if let Some(ref pose) = last_kf.pose {
                let relative = pose.inverse().compose(&self.cumulative_pose);
                let translation = relative.translation.norm();
                let rotation = rotation_angle(&relative.rotation);

                println!(
                    "Motion since last KF: trans={:.3}, rot={:.3}",
                    translation, rotation
                );

                // Create keyframe if significant motion
                if translation > 0.15 || rotation > 0.15 {
                    return true;
                }
            } else {
                // Handle the case where pose is None
                println!("Last keyframe does not have a pose.");
            }
        }

        false
    }

    fn create_keyframe(&mut self, frame: Frame, pose: SE3) -> TrackerResult<()> {
        println!("Creating new keyframe {} with {} points in map", 
                 frame.id, self.local_map.get_point_count());

        let pyramid = self.pyramid.as_ref().unwrap();
        let mut keyframe = frame.clone();
        keyframe.pose = Some(pose.clone());

        // Always add the new keyframe to the map
        self.local_map.add_keyframe(keyframe.id, pose.clone());

        // Triangulate with the last keyframe if it exists
        if let Some(last_kf) = &self.last_keyframe {
            let matches = self.matcher.match_descriptors(
                &last_kf.descriptors,
                &keyframe.descriptors,
                &pyramid.scale_factors,
            );

            println!("Found {} matches with last keyframe {}", matches.len(), last_kf.id);

            if matches.len() >= MIN_TRACKED_POINTS {
                self.local_map.triangulate_points(
                    last_kf.id,
                    keyframe.id,
                    &matches,
                    last_kf,
                    &keyframe,
                )?;

                println!("Map size after triangulation: {}", self.local_map.get_point_count());
            }
        }

        // Update keyframe state
        self.last_keyframe = Some(keyframe);
        self.frames_since_keyframe = 0;

        Ok(())
    }

    fn recover_tracking(&mut self, frame: &Frame) -> TrackerResult<SE3> {
        println!("Attempting to recover tracking for frame {}", frame.id);

        let pyramid = self.pyramid.as_ref().unwrap();
        // First try: Match with last keyframe
        if let Some(last_kf) = &self.last_keyframe {
            println!("Attempting recovery using last keyframe {}", last_kf.id);

            // Try matching with wider threshold
            let matches = self.matcher.match_descriptors(
                &last_kf.descriptors,
                &frame.descriptors,
                &pyramid.scale_factors,
            );

            if matches.len() >= MIN_TRACKED_POINTS {
                // Get visible map points from last known pose
                let visible_points = self
                    .local_map
                    .get_visible_points(&self.cumulative_pose, None);

                if visible_points.len() >= MIN_MAP_POINTS {
                    let points_3d: Vec<_> = visible_points.iter().map(|(p, _)| p.position).collect();
                    let points_2d: Vec<_> = visible_points
                        .iter()
                        .map(|(_, proj)| proj.clone())
                        .collect();

                    // Try PnP with relaxed thresholds
                    if let Ok(pose_result) = self.pose_estimator.estimate_pose_pnp(
                        &points_3d,
                        &points_2d,
                        Some(&self.cumulative_pose),
                    ) {
                        println!("Recovery successful using PnP");
                        self.tracking_state = TrackingState::Tracking;
                        self.last_frame = Some(frame.clone());
                        self.last_pose_change = None; // Reset motion model
                        return Ok(pose_result.pose);
                    }
                }

                // Fallback to essential matrix
                if let Ok(pose_result) = self.estimate_pose_essential(
                    &matches,
                    last_kf,
                    &frame,
                    Some(&self.cumulative_pose),
                ) {
                    println!("Recovery successful using essential matrix");
                    self.tracking_state = TrackingState::Tracking;
                    self.last_frame = Some(frame.clone());
                    self.last_pose_change = None;
                    return Ok(pose_result.pose);
                }
            }
        }

        // If recovery failed, reset to initialization
        println!("Recovery failed, resetting to initialization");
        self.tracking_state = TrackingState::Initializing;
        self.last_pose_change = None;
        self.scale_history.clear();
        self.last_frame = Some(frame.clone());

        Ok(SE3::identity())
    }

    pub fn get_tracking_stats(&self) -> String {
        format!(
            "State: {:?}\nTracked Points: {}\nQuality: {:.2}\nMap Points: {}\nScale: {:.3}",
            self.tracking_state,
            self.num_tracked_points,
            self.tracking_quality,
            self.local_map.get_point_count(),
            self.scale_history.back().unwrap_or(&1.0)
        )
    }
}

// Helper function for rotation angle
fn rotation_angle(r: &na::Matrix3<f64>) -> f64 {
    let trace = r.trace();
    let cos_theta = (trace - 1.0) / 2.0;
    cos_theta.clamp(-1.0, 1.0).acos()
}

fn compute_median_parallax(points: &[na::Vector3<f64>]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    let mut angles = Vec::with_capacity(points.len());
    let reference = na::Vector3::new(0.0, 0.0, 1.0); // Forward direction

    for point in points {
        let angle = reference.angle(point);
        angles.push(angle.to_degrees());
    }

    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    angles[angles.len() / 2]
}

fn convert_descriptors_to_features(descriptors: &[Vec<OrbDescriptor>]) -> Vec<Feature> {
    descriptors
        .iter()
        .flat_map(|level_desc| {
            level_desc.iter().map(|desc| Feature {
                point: na::Point2::new(desc.keypoint.pt.0 as f64, desc.keypoint.pt.1 as f64),
                octave: desc.keypoint.octave,
                angle: desc.keypoint.angle,
                response: desc.keypoint.response,
            })
        })
        .collect()
}

fn compute_viewing_angle(pose: &SE3, point: &na::Point3<f64>) -> f64 {
    let view_dir = (point - pose.translation).coords.normalize();
    let camera_dir = pose.rotation * na::Vector3::new(0.0, 0.0, 1.0);
    view_dir.angle(&camera_dir).to_degrees()
}
