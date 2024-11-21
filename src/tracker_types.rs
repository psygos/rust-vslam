use crate::new_distributed_extraction::OrbDescriptor;
use crate::pose_estimator::CameraIntrinsics;
use nalgebra as na;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct SE3 {
    pub rotation: na::Matrix3<f64>,
    pub translation: na::Vector3<f64>,
}

impl SE3 {
    pub fn identity() -> Self {
        Self {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::zeros(),
        }
    }

    pub fn compose(&self, other: &SE3) -> SE3 {
        SE3 {
            rotation: self.rotation * other.rotation,
            translation: self.rotation * other.translation + self.translation,
        }
    }

    pub fn inverse(&self) -> SE3 {
        let rot_inv = self.rotation.transpose();
        SE3 {
            rotation: rot_inv,
            translation: -rot_inv * self.translation,
        }
    }
    pub fn from_translation(t: na::Vector3<f64>) -> Self {
        Self {
            rotation: na::Matrix3::identity(),
            translation: t,
        } 
}
}
impl Default for SE3 {
    fn default() -> Self {
        SE3 {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::zeros(),
        }
    }
}


#[derive(Debug, Clone)]
pub struct Feature {
    pub point: na::Point2<f64>,
    pub octave: i32,
    pub angle: f32,
    pub response: f32,
}
#[derive(Debug, Clone)]
pub struct TrackedFeature {
    pub feature: Feature,
    pub id: usize,
    pub first_frame_id: usize,
    pub last_seen_frame_id: usize,
    pub positions: Vec<na::Point2<f64>>,
    pub descriptor: Vec<u8>,
    pub keyframe_id: Option<usize>,
    pub keyframe_position: Option<na::Point2<f64>>,
}
#[derive(Debug, Clone)]
pub struct Frame {
    pub id: usize,
    pub timestamp: f64,
    pub features: Vec<Feature>,
    pub descriptors: Vec<Vec<OrbDescriptor>>,
    pub camera: Arc<CameraIntrinsics>,
    pub tracked_features: Option<Vec<TrackedFeature>>,
    pub pose: Option<SE3>,
}

impl Frame {
    pub fn new(
        id: usize,
        timestamp: f64,
        features: Vec<Feature>,
        descriptors: Vec<Vec<OrbDescriptor>>,
        camera: Arc<CameraIntrinsics>,
        tracked_features: Option<Vec<TrackedFeature>>,
    ) -> Self {
        Self {
            id,
            timestamp,
            features,
            descriptors,
            camera,
            tracked_features,
            pose: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RansacParams {
    pub min_inliers: usize,
    pub max_iterations: usize,
    pub threshold: f64,
    pub confidence: f64,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            min_inliers: 30,
            max_iterations: 200,
            threshold: 0.01,
            confidence: 0.999,
        }
    }
}

#[derive(Error, Debug)]
pub enum TrackerError {
    #[error("Insufficient matches for pose estimation: {0} < {1}")]
    InsufficientMatches(usize, usize),

    #[error("Failed to estimate pose: {0}")]
    PoseEstimationFailed(String),

    #[error("Lost tracking, confidence too low: {0} < {1}")]
    LowTrackingConfidence(f64, f64),

    #[error("Failed to initialize: {0}")]
    InitializationFailed(String),

    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),
}

pub type TrackerResult<T> = Result<T, TrackerError>;
