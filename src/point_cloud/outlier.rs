use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};

impl HighPerformancePointCloud {
    pub fn remove_statistical_outlier(
        &self,
        nb_neighbors: usize,
        std_ratio: f32,
    ) -> Result<(Self, Vec<bool>)> {
        if self.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot remove outliers from an empty point cloud".to_string(),
            ));
        }
        if nb_neighbors < 2 {
            return Err(PointCloudError::InvalidParameter(
                "nb_neighbors must be >= 2".to_string(),
            ));
        }
        if !std_ratio.is_finite() || std_ratio < 0.0 {
            return Err(PointCloudError::InvalidParameter(
                "std_ratio must be finite and non-negative".to_string(),
            ));
        }

        let xyz = self.get_xyz_vec();
        let hits = self
            .kdtree()?
            .knn(&xyz, (nb_neighbors + 1).min(self.point_count()))?;
        let mean_distances: Vec<f32> = hits
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let distances: Vec<f32> = row
                    .into_iter()
                    .filter(|hit| hit.index as usize != i)
                    .take(nb_neighbors)
                    .map(|hit| hit.distance)
                    .collect();
                if distances.is_empty() {
                    0.0
                } else {
                    distances.iter().sum::<f32>() / distances.len() as f32
                }
            })
            .collect();

        let mean = mean_distances.iter().sum::<f32>() / mean_distances.len() as f32;
        let variance = mean_distances
            .iter()
            .map(|d| {
                let delta = d - mean;
                delta * delta
            })
            .sum::<f32>()
            / mean_distances.len() as f32;
        let threshold = mean + std_ratio * variance.sqrt();
        let kept_mask: Vec<bool> = mean_distances.iter().map(|&d| d <= threshold).collect();
        let filtered = self.select_mask(&kept_mask)?;
        Ok((filtered, kept_mask))
    }

    pub fn remove_radius_outlier(
        &self,
        nb_points: usize,
        radius: f32,
    ) -> Result<(Self, Vec<bool>)> {
        if self.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot remove outliers from an empty point cloud".to_string(),
            ));
        }
        if nb_points == 0 {
            return Err(PointCloudError::InvalidParameter(
                "nb_points must be greater than zero".to_string(),
            ));
        }
        let xyz = self.get_xyz_vec();
        let hits = self.kdtree()?.radius_search(&xyz, radius)?;
        let kept_mask: Vec<bool> = hits
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                row.into_iter()
                    .filter(|hit| hit.index as usize != i)
                    .take(nb_points)
                    .count()
                    >= nb_points
            })
            .collect();
        let filtered = self.select_mask(&kept_mask)?;
        Ok((filtered, kept_mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_outlier_removes_isolated_point() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        .unwrap();
        let (filtered, mask) = pc.remove_radius_outlier(1, 1.5).unwrap();
        assert_eq!(filtered.point_count(), 3);
        assert_eq!(mask, vec![true, true, true, false]);
    }

    #[test]
    fn statistical_outlier_removes_far_point() {
        let mut xyz = Vec::new();
        for i in 0..40 {
            xyz.push([i as f32 * 0.01, 0.0, 0.0]);
        }
        xyz.push([50.0, 50.0, 50.0]);
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let (_filtered, mask) = pc.remove_statistical_outlier(4, 1.0).unwrap();
        assert!(!mask[40]);
    }
}
