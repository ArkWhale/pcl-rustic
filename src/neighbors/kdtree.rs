use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use kiddo::{KdTree, SquaredEuclidean};

#[derive(Clone, Debug)]
pub struct NeighborHit {
    pub index: u64,
    pub distance: f32,
}

pub struct KdTreeIndex {
    inner: Option<KdTree<f32, 3>>,
    xyz_host: Vec<[f32; 3]>,
}

impl KdTreeIndex {
    pub fn build(pc: &HighPerformancePointCloud) -> Result<Self> {
        let xyz_host = pc.get_xyz_vec();
        if xyz_host.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot build KD-tree for an empty point cloud".to_string(),
            ));
        }
        if xyz_host.iter().flatten().any(|v| !v.is_finite()) {
            return Err(PointCloudError::InvalidParameter(
                "cannot build KD-tree with non-finite coordinates".to_string(),
            ));
        }

        let inner = if has_axis_bucket_over_limit(&xyz_host, 32) {
            log::warn!(
                "degenerate geometry exceeds kiddo bucket limits; using brute-force queries"
            );
            None
        } else {
            let mut inner: KdTree<f32, 3> = KdTree::new();
            for (idx, point) in xyz_host.iter().enumerate() {
                inner.add(point, idx as u64);
            }
            Some(inner)
        };
        Ok(Self { inner, xyz_host })
    }

    pub fn point_count(&self) -> usize {
        self.xyz_host.len()
    }

    pub fn xyz(&self) -> &[[f32; 3]] {
        &self.xyz_host
    }

    pub fn knn(&self, query: &[[f32; 3]], k: usize) -> Result<Vec<Vec<NeighborHit>>> {
        if k == 0 {
            return Err(PointCloudError::InvalidParameter(
                "k must be greater than zero".to_string(),
            ));
        }
        let k = k.min(self.xyz_host.len());
        if let Some(inner) = &self.inner {
            Ok(query
                .iter()
                .map(|point| {
                    inner
                        .nearest_n::<SquaredEuclidean>(point, k)
                        .into_iter()
                        .map(|hit| NeighborHit {
                            index: hit.item,
                            distance: hit.distance.sqrt(),
                        })
                        .collect()
                })
                .collect())
        } else {
            Ok(query
                .iter()
                .map(|point| brute_force_knn(&self.xyz_host, point, k))
                .collect())
        }
    }

    pub fn radius_search(&self, query: &[[f32; 3]], radius: f32) -> Result<Vec<Vec<NeighborHit>>> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(PointCloudError::InvalidParameter(
                "radius must be finite and non-negative".to_string(),
            ));
        }
        let radius_sq = radius * radius;
        if let Some(inner) = &self.inner {
            Ok(query
                .iter()
                .map(|point| {
                    inner
                        .within::<SquaredEuclidean>(point, radius_sq)
                        .into_iter()
                        .map(|hit| NeighborHit {
                            index: hit.item,
                            distance: hit.distance.sqrt(),
                        })
                        .collect()
                })
                .collect())
        } else {
            Ok(query
                .iter()
                .map(|point| brute_force_radius(&self.xyz_host, point, radius))
                .collect())
        }
    }
}

fn brute_force_knn(xyz: &[[f32; 3]], query: &[f32; 3], k: usize) -> Vec<NeighborHit> {
    let mut hits: Vec<NeighborHit> = xyz
        .iter()
        .enumerate()
        .map(|(idx, point)| NeighborHit {
            index: idx as u64,
            distance: distance(point, query),
        })
        .collect();
    hits.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(k);
    hits
}

fn brute_force_radius(xyz: &[[f32; 3]], query: &[f32; 3], radius: f32) -> Vec<NeighborHit> {
    let mut hits: Vec<NeighborHit> = xyz
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| {
            let d = distance(point, query);
            (d <= radius).then_some(NeighborHit {
                index: idx as u64,
                distance: d,
            })
        })
        .collect();
    hits.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn has_axis_bucket_over_limit(xyz: &[[f32; 3]], limit: usize) -> bool {
    for axis in 0..3 {
        let mut values: Vec<u32> = xyz.iter().map(|p| p[axis].to_bits()).collect();
        values.sort_unstable();
        let mut run = 1usize;
        for pair in values.windows(2) {
            if pair[0] == pair[1] {
                run += 1;
                if run > limit {
                    return true;
                }
            } else {
                run = 1;
            }
        }
    }
    false
}

impl HighPerformancePointCloud {
    pub fn knn(&self, query: &[[f32; 3]], k: usize) -> Result<Vec<Vec<NeighborHit>>> {
        self.kdtree()?.knn(query, k)
    }

    pub fn radius_search(&self, query: &[[f32; 3]], radius: f32) -> Result<Vec<Vec<NeighborHit>>> {
        self.kdtree()?.radius_search(query, radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tensor;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn knn_matches_expected_order() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        .unwrap();
        let hits = pc.knn(&[[0.1, 0.0, 0.0]], 2).unwrap();
        assert_eq!(hits[0][0].index, 0);
        assert_eq!(hits[0][1].index, 1);
    }

    #[test]
    fn radius_search_matches_expected_set() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        .unwrap();
        let mut ids: Vec<u64> = pc.radius_search(&[[0.0, 0.0, 0.0]], 1.1).unwrap()[0]
            .iter()
            .map(|hit| hit.index)
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn degenerate_axis_uses_deterministic_fallback() {
        let xyz: Vec<[f32; 3]> = (0..40).map(|i| [0.0, i as f32, 0.0]).collect();
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();
        assert!(index.inner.is_none());

        let query = [[0.0, 10.2, 0.0], [0.0, 25.8, 0.0]];
        let first_knn = index.knn(&query, 3).unwrap();
        let second_knn = index.knn(&query, 3).unwrap();
        assert_eq!(hit_ids(&first_knn), hit_ids(&second_knn));
        assert_eq!(hit_ids(&first_knn), vec![vec![10, 11, 9], vec![26, 25, 27]]);

        let first_radius = index.radius_search(&query, 1.25).unwrap();
        let second_radius = index.radius_search(&query, 1.25).unwrap();
        assert_eq!(hit_ids(&first_radius), hit_ids(&second_radius));
        assert_eq!(
            hit_ids(&first_radius),
            vec![vec![10, 11, 9], vec![26, 25, 27]]
        );
    }

    #[test]
    fn random_10k_queries_match_bruteforce_oracle() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let xyz: Vec<[f32; 3]> = (0..10_000)
            .map(|_| {
                [
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ]
            })
            .collect();
        let query: Vec<[f32; 3]> = (0..128)
            .map(|_| {
                [
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                    rng.gen_range(-1000.0..1000.0),
                ]
            })
            .collect();
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz.clone()).unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();

        let knn = index.knn(&query, 8).unwrap();
        for (actual, point) in knn.iter().zip(query.iter()) {
            let expected = brute_force_knn(&xyz, point, 8);
            assert_eq!(
                actual.iter().map(|hit| hit.index).collect::<Vec<_>>(),
                expected.iter().map(|hit| hit.index).collect::<Vec<_>>()
            );
        }

        let radius = 125.0;
        let radius_hits = index.radius_search(&query, radius).unwrap();
        for (actual, point) in radius_hits.iter().zip(query.iter()) {
            let mut actual_ids: Vec<u64> = actual.iter().map(|hit| hit.index).collect();
            let mut expected_ids: Vec<u64> = brute_force_radius(&xyz, point, radius)
                .iter()
                .map(|hit| hit.index)
                .collect();
            actual_ids.sort_unstable();
            expected_ids.sort_unstable();
            assert_eq!(actual_ids, expected_ids);
        }
    }

    #[test]
    fn empty_and_non_finite_clouds_return_errors() {
        let empty = HighPerformancePointCloud::new();
        match empty.kdtree() {
            Ok(_) => panic!("empty cloud unexpectedly built a KD-tree"),
            Err(err) => assert!(err.to_string().contains("empty")),
        }

        let non_finite =
            HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 0.0], [f32::NAN, 1.0, 2.0]])
                .unwrap();
        match non_finite.kdtree() {
            Ok(_) => panic!("non-finite cloud unexpectedly built a KD-tree"),
            Err(err) => assert!(err.to_string().contains("non-finite")),
        }
    }

    #[test]
    fn repeated_kdtree_access_reuses_cached_index() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        .unwrap();

        let first = pc.kdtree().unwrap() as *const KdTreeIndex;
        let second = pc.kdtree().unwrap() as *const KdTreeIndex;

        assert_eq!(first, second);
    }

    #[test]
    fn xyz_mut_invalidates_cached_kdtree() {
        let mut pc =
            HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
                .unwrap();
        assert_eq!(pc.knn(&[[5.1, 0.0, 0.0]], 1).unwrap()[0][0].index, 1);

        *pc.xyz_mut() = tensor::tensor2_from_slice(&[5.0, 0.0, 0.0, 6.0, 0.0, 0.0], 2, 3).unwrap();

        assert_eq!(pc.knn(&[[5.1, 0.0, 0.0]], 1).unwrap()[0][0].index, 0);
    }

    fn hit_ids(rows: &[Vec<NeighborHit>]) -> Vec<Vec<u64>> {
        rows.iter()
            .map(|row| row.iter().map(|hit| hit.index).collect())
            .collect()
    }
}
