use crate::feature_matcher::FeatureMatch;
use crate::new_distributed_extraction::OrbDescriptor;
use crate::tracker_types::{Feature, Frame, TrackedFeature, TrackerError, TrackerResult};
use nalgebra as na;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Represents a single cell in the image grid for feature distribution
#[derive(Debug)]
struct GridCell {
    features: HashSet<usize>,
}

impl GridCell {
    fn new() -> Self {
        Self {
            features: HashSet::new(),
        }
    }

    fn clear(&mut self) {
        self.features.clear();
    }

    fn add_feature(&mut self, feature_id: usize) -> bool {
        self.features.insert(feature_id)
    }

    fn count(&self) -> usize {
        self.features.len()
    }
}

#[derive(Debug, Clone)]
pub struct FeatureManagerConfig {
    pub grid_size: (usize, usize),
    pub max_features_per_cell: usize,
    pub min_tracking_length: usize,
    pub max_tracking_length: usize,
    pub max_frames_to_keep_lost: usize,
}

impl Default for FeatureManagerConfig {
    fn default() -> Self {
        Self {
            grid_size: (4, 4),
            max_features_per_cell: 50,
            min_tracking_length: 3,
            max_tracking_length: 30,
            max_frames_to_keep_lost: 2,
        }
    }
}

#[derive(Debug)]
pub struct FeatureManager {
    config: FeatureManagerConfig,
    next_feature_id: AtomicUsize,
    tracked_features: HashMap<usize, TrackedFeature>,
    grid_cells: Vec<Vec<GridCell>>,
    current_keyframe_id: Option<usize>,
}

impl FeatureManager {
    pub fn new(grid_size: (usize, usize), max_features_per_cell: usize) -> Self {
        let config = FeatureManagerConfig {
            grid_size,
            max_features_per_cell,
            ..Default::default()
        };

        let mut grid_cells = Vec::with_capacity(config.grid_size.1);
        for _ in 0..config.grid_size.1 {
            let mut row = Vec::with_capacity(config.grid_size.0);
            for _ in 0..config.grid_size.0 {
                row.push(GridCell::new());
            }
            grid_cells.push(row);
        }

        Self {
            config,
            next_feature_id: AtomicUsize::new(0),
            tracked_features: HashMap::new(),
            grid_cells,
            current_keyframe_id: None,
        }
    }

    fn get_cell_indices(
        &self,
        point: &na::Point2<f64>,
        width: u32,
        height: u32,
    ) -> Option<(usize, usize)> {
        if width == 0 || height == 0 {
            return None;
        }

        let x_cell = ((point.x / width as f64) * self.config.grid_size.0 as f64).floor() as usize;
        let y_cell = ((point.y / height as f64) * self.config.grid_size.1 as f64).floor() as usize;

        if x_cell >= self.config.grid_size.0 || y_cell >= self.config.grid_size.1 {
            None
        } else {
            Some((x_cell, y_cell))
        }
    }

    fn create_new_track(
        &self,
        id: usize,
        curr_feature: &Feature,
        prev_feat: &na::Point2<f64>,
        curr_feat: &na::Point2<f64>,
        curr_descriptor: &[u8],
        prev_frame_id: usize,
        curr_frame_id: usize,
    ) -> TrackedFeature {
        let mut track = TrackedFeature {
            feature: curr_feature.clone(),
            id,
            first_frame_id: prev_frame_id,
            last_seen_frame_id: curr_frame_id,
            positions: vec![prev_feat.clone(), curr_feat.clone()],
            descriptor: curr_descriptor.to_vec(),
            keyframe_id: self.current_keyframe_id,
            keyframe_position: if self.current_keyframe_id.is_some() {
                Some(prev_feat.clone())
            } else {
                None
            },
        };

        // If we have a current keyframe, set its position as the keyframe reference
        if let Some(kf_id) = self.current_keyframe_id {
            if prev_frame_id == kf_id {
                track.keyframe_position = Some(prev_feat.clone());
            }
        }

        track
    }

    fn update_existing_track(
        track: &mut TrackedFeature,
        curr_feature: &Feature,
        curr_feat: &na::Point2<f64>,
        curr_frame_id: usize,
        curr_descriptor: &[u8],
    ) {
        track.feature = curr_feature.clone();
        track.positions.push(curr_feat.clone());
        track.last_seen_frame_id = curr_frame_id;
        track.descriptor = curr_descriptor.to_vec();
        // Note: We don't update keyframe information here as it should stay fixed
    }

    pub fn set_current_keyframe(&mut self, frame_id: Option<usize>) {
        self.current_keyframe_id = frame_id;
    }

    pub fn track_features(
        &mut self,
        prev_frame: &Frame,
        curr_frame: &Frame,
        matches: &[FeatureMatch],
        curr_descriptors: &[Vec<OrbDescriptor>],
    ) -> TrackerResult<Vec<TrackedFeature>> {
        // Initial validation
        if curr_frame.camera.width == 0 || curr_frame.camera.height == 0 {
            return Err(TrackerError::PoseEstimationFailed(
                "Invalid camera dimensions".to_string(),
            ));
        }

        // Clear grid but keep track of grid occupancy without enforcing limits yet
        for row in &mut self.grid_cells {
            for cell in row {
                cell.clear();
            }
        }

        let mut match_scores: Vec<(usize, f64, FeatureMatch)> = matches
            .iter()
            .filter_map(|m| {
                let prev_point = prev_frame.features[m.query_idx].point;

                // Find existing track
                let track_length = self
                    .tracked_features
                    .values()
                    .find(|t| {
                        t.last_seen_frame_id == prev_frame.id
                            && t.positions.last() == Some(&prev_point)
                    })
                    .map_or(1, |t| t.positions.len());

                // Scoring that heavily favors long tracks
                let length_score = (track_length as f64).min(10.0) / 10.0; // Normalize to [0,1]
                let distance_score = 1.0 / (m.distance as f64 + 1.0);
                let score = length_score * 0.7 + distance_score * 0.3; // Prioritize track length

                Some((m.train_idx, score, m.clone()))
            })
            .collect();

        // Sort by score descending
        match_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut active_tracks = Vec::new();
        let mut updated_features = HashSet::new();

        // Second pass: Process matches in order of importance
        for (_, _, m) in match_scores {
            let prev_point = prev_frame.features[m.query_idx].point;
            let curr_point = curr_frame.features[m.train_idx].point;
            let curr_feature = &curr_frame.features[m.train_idx];

            let cell_indices = match self.get_cell_indices(
                &curr_point,
                curr_frame.camera.width,
                curr_frame.camera.height,
            ) {
                Some(indices) => indices,
                None => continue,
            };

            // Check grid occupancy but allow overflow for high-quality tracks
            let track_length = self
                .tracked_features
                .values()
                .find(|t| {
                    t.last_seen_frame_id == prev_frame.id && t.positions.last() == Some(&prev_point)
                })
                .map_or(1, |t| t.positions.len());

            let cell_count = self.grid_cells[cell_indices.1][cell_indices.0].count();
            if track_length < 5 && cell_count >= self.config.max_features_per_cell * 2 {
                continue;
            }

            let track = if let Some(existing_track) = self.tracked_features.values_mut().find(|t| {
                t.last_seen_frame_id == prev_frame.id
                    && !updated_features.contains(&t.id)
                    && t.positions.last() == Some(&prev_point)
            }) {
                updated_features.insert(existing_track.id);
                Self::update_existing_track(
                    existing_track,
                    curr_feature,
                    &curr_point,
                    curr_frame.id,
                    &curr_descriptors[m.train_level][m.train_idx].descriptor,
                );
                existing_track.clone()
            } else {
                let id = self.next_feature_id.fetch_add(1, Ordering::SeqCst);
                let track = self.create_new_track(
                    id,
                    curr_feature,
                    &prev_point,
                    &curr_point,
                    &curr_descriptors[m.train_level][m.train_idx].descriptor,
                    prev_frame.id,
                    curr_frame.id,
                );
                self.tracked_features.insert(id, track.clone());
                track
            };

            self.grid_cells[cell_indices.1][cell_indices.0].add_feature(track.id);
            active_tracks.push(track);
        }

        // More lenient cleanup
        self.tracked_features.retain(|_, track| {
            curr_frame.id - track.last_seen_frame_id < self.config.max_frames_to_keep_lost * 2
                || track.positions.len() >= self.config.min_tracking_length
                || track.keyframe_id.is_some()
        });

        active_tracks.sort_by_key(|t| -(t.positions.len() as i32));
        Ok(active_tracks)
    }

    pub fn get_keyframe_tracks(&self, keyframe_id: usize) -> Vec<&TrackedFeature> {
        self.tracked_features
            .values()
            .filter(|t| t.keyframe_id == Some(keyframe_id))
            .collect()
    }

    pub fn get_long_tracks(&self) -> Vec<&TrackedFeature> {
        self.tracked_features
            .values()
            .filter(|t| t.positions.len() >= self.config.min_tracking_length)
            .collect()
    }

    pub fn get_track_by_id(&self, id: usize) -> Option<&TrackedFeature> {
        self.tracked_features.get(&id)
    }

    pub fn get_active_tracks(&self, curr_frame_id: usize) -> Vec<&TrackedFeature> {
        self.tracked_features
            .values()
            .filter(|t| t.last_seen_frame_id == curr_frame_id)
            .collect()
    }

    pub fn get_stats(&self) -> FeatureManagerStats {
        FeatureManagerStats {
            total_tracks: self.tracked_features.len(),
            long_tracks: self.get_long_tracks().len(),
            grid_distribution: self
                .grid_cells
                .iter()
                .map(|row| row.iter().map(|cell| cell.count()).collect())
                .collect(),
        }
    }
}

#[derive(Debug)]
pub struct FeatureManagerStats {
    pub total_tracks: usize,
    pub long_tracks: usize,
    pub grid_distribution: Vec<Vec<usize>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_indices() {
        let manager = FeatureManager::new((4, 4), 50);

        // Test point in first cell
        let point1 = na::Point2::new(10.0, 10.0);
        assert_eq!(manager.get_cell_indices(&point1, 100, 100), Some((0, 0)));

        // Test point in last cell
        let point2 = na::Point2::new(99.0, 99.0);
        assert_eq!(manager.get_cell_indices(&point2, 100, 100), Some((3, 3)));

        // Test invalid dimensions
        assert_eq!(manager.get_cell_indices(&point1, 0, 100), None);
    }
}
