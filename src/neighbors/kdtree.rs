use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use kiddo::{float::kdtree::KdTree as FloatKdTree, KdTree, SquaredEuclidean};
use rayon::prelude::*;

const PROJECTED_BUCKET_SIZE: usize = 4096;
type LargeKdTree3d = FloatKdTree<f32, u64, 3, PROJECTED_BUCKET_SIZE, u32>;
type ProjectedKdTree2d = FloatKdTree<f32, u64, 2, PROJECTED_BUCKET_SIZE, u32>;
type ProjectedKdTree1d = FloatKdTree<f32, u64, 1, PROJECTED_BUCKET_SIZE, u32>;

#[derive(Clone, Debug)]
pub struct NeighborHit {
    pub index: u64,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KnnHit {
    pub index: u64,
    pub distance_sq: f32,
}

enum KdTreeBackend {
    Kd3d(KdTree<f32, 3>),
    LargeKd3d(LargeKdTree3d),
    Projected2d {
        tree: ProjectedKdTree2d,
        axes: [usize; 2],
    },
    Projected1d {
        tree: ProjectedKdTree1d,
        axis: usize,
    },
    BruteForce,
}

pub struct KdTreeIndex {
    backend: KdTreeBackend,
    xyz_host: Vec<[f32; 3]>,
}

impl KdTreeIndex {
    pub fn build(pc: &HighPerformancePointCloud) -> Result<Self> {
        let xyz_host = pc.get_xyz_vec();
        Self::build_from_xyz(xyz_host)
    }

    pub(crate) fn build_from_xyz(xyz_host: Vec<[f32; 3]>) -> Result<Self> {
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

        let backend = build_backend(&xyz_host);
        Ok(Self { backend, xyz_host })
    }

    pub(crate) fn xyz(&self) -> &[[f32; 3]] {
        &self.xyz_host
    }

    pub fn knn(&self, query: &[[f32; 3]], k: usize) -> Result<Vec<Vec<NeighborHit>>> {
        if k == 0 {
            return Err(PointCloudError::InvalidParameter(
                "k must be greater than zero".to_string(),
            ));
        }
        let k = k.min(self.xyz_host.len());
        Ok(query
            .par_iter()
            .map(|point| self.knn_point(point, k))
            .collect())
    }

    pub(crate) fn knn_one_within(
        &self,
        query: &[[f32; 3]],
        max_distance_sq: f32,
    ) -> Result<Vec<Option<KnnHit>>> {
        if max_distance_sq < 0.0 || !max_distance_sq.is_finite() {
            return Err(PointCloudError::InvalidParameter(
                "max_distance_sq must be finite and non-negative".to_string(),
            ));
        }
        Ok(query
            .par_iter()
            .map(|point| self.knn_one_within_point(point, max_distance_sq))
            .collect())
    }

    pub(crate) fn knn_mean_distances_for_index_points(
        &self,
        k_without_self: usize,
    ) -> Result<Vec<f32>> {
        if k_without_self == 0 {
            return Err(PointCloudError::InvalidParameter(
                "k_without_self must be greater than zero".to_string(),
            ));
        }
        let k = k_without_self.saturating_add(1).min(self.xyz_host.len());
        Ok((0..self.xyz_host.len())
            .into_par_iter()
            .map(|idx| self.mean_knn_distance_for_index_point(idx, k_without_self, k))
            .collect())
    }

    pub(crate) fn radius_self_has_at_least(
        &self,
        radius: f32,
        min_neighbors_without_self: usize,
    ) -> Result<Vec<bool>> {
        if min_neighbors_without_self == 0 {
            return Err(PointCloudError::InvalidParameter(
                "min_neighbors_without_self must be greater than zero".to_string(),
            ));
        }
        if radius < 0.0 || !radius.is_finite() {
            return Err(PointCloudError::InvalidParameter(
                "radius must be finite and non-negative".to_string(),
            ));
        }
        let radius_sq = radius * radius;
        Ok((0..self.xyz_host.len())
            .into_par_iter()
            .map(|idx| {
                self.radius_count_reaches_for_index_point(
                    idx,
                    radius_sq,
                    min_neighbors_without_self,
                )
            })
            .collect())
    }

    fn knn_one_within_point(&self, point: &[f32; 3], max_distance_sq: f32) -> Option<KnnHit> {
        let hit = self.knn_one_point(point)?;
        (hit.distance_sq <= max_distance_sq).then_some(hit)
    }

    fn mean_knn_distance_for_index_point(
        &self,
        point_index: usize,
        k_without_self: usize,
        k_with_self: usize,
    ) -> f32 {
        let distances: Vec<f32> = self
            .knn_point(&self.xyz_host[point_index], k_with_self)
            .into_iter()
            .filter(|hit| hit.index as usize != point_index)
            .take(k_without_self)
            .map(|hit| hit.distance)
            .collect();
        if distances.is_empty() {
            0.0
        } else {
            distances.iter().sum::<f32>() / distances.len() as f32
        }
    }

    fn knn_point(&self, point: &[f32; 3], k: usize) -> Vec<NeighborHit> {
        match &self.backend {
            KdTreeBackend::Kd3d(tree) => tree
                .nearest_n::<SquaredEuclidean>(point, k)
                .into_iter()
                .map(|hit| NeighborHit {
                    index: hit.item,
                    distance: hit.distance.sqrt(),
                })
                .collect(),
            KdTreeBackend::LargeKd3d(tree) => tree
                .nearest_n::<SquaredEuclidean>(point, k)
                .into_iter()
                .map(|hit| NeighborHit {
                    index: hit.item,
                    distance: hit.distance.sqrt(),
                })
                .collect(),
            KdTreeBackend::Projected2d { tree, axes } => tree
                .nearest_n::<SquaredEuclidean>(&project2(point, *axes), k)
                .into_iter()
                .map(|hit| {
                    let distance = distance(&self.xyz_host[hit.item as usize], point);
                    NeighborHit {
                        index: hit.item,
                        distance,
                    }
                })
                .collect(),
            KdTreeBackend::Projected1d { tree, axis } => tree
                .nearest_n::<SquaredEuclidean>(&project1(point, *axis), k)
                .into_iter()
                .map(|hit| {
                    let distance = distance(&self.xyz_host[hit.item as usize], point);
                    NeighborHit {
                        index: hit.item,
                        distance,
                    }
                })
                .collect(),
            KdTreeBackend::BruteForce => brute_force_knn(&self.xyz_host, point, k),
        }
    }

    fn knn_one_point(&self, point: &[f32; 3]) -> Option<KnnHit> {
        match &self.backend {
            KdTreeBackend::Kd3d(tree) => {
                let hit = tree.nearest_one::<SquaredEuclidean>(point);
                Some(KnnHit {
                    index: hit.item,
                    distance_sq: hit.distance,
                })
            }
            KdTreeBackend::LargeKd3d(tree) => {
                let hit = tree.nearest_one::<SquaredEuclidean>(point);
                Some(KnnHit {
                    index: hit.item,
                    distance_sq: hit.distance,
                })
            }
            KdTreeBackend::Projected2d { tree, axes } => {
                let hit = tree.nearest_one::<SquaredEuclidean>(&project2(point, *axes));
                Some(KnnHit {
                    index: hit.item,
                    distance_sq: squared_distance(&self.xyz_host[hit.item as usize], point),
                })
            }
            KdTreeBackend::Projected1d { tree, axis } => {
                let hit = tree.nearest_one::<SquaredEuclidean>(&project1(point, *axis));
                Some(KnnHit {
                    index: hit.item,
                    distance_sq: squared_distance(&self.xyz_host[hit.item as usize], point),
                })
            }
            KdTreeBackend::BruteForce => brute_force_nearest_one(&self.xyz_host, point),
        }
    }

    fn radius_count_reaches_for_index_point(
        &self,
        point_index: usize,
        radius_sq: f32,
        min_neighbors_without_self: usize,
    ) -> bool {
        let mut count = 0usize;
        let point = &self.xyz_host[point_index];

        let mut reaches_count = |idx: u64| {
            if idx as usize != point_index {
                count += 1;
                if count >= min_neighbors_without_self {
                    return true;
                }
            }
            false
        };

        match &self.backend {
            KdTreeBackend::Kd3d(tree) => {
                for hit in tree.within_unsorted::<SquaredEuclidean>(point, radius_sq) {
                    if reaches_count(hit.item) {
                        return true;
                    }
                }
            }
            KdTreeBackend::LargeKd3d(tree) => {
                for hit in tree.within_unsorted::<SquaredEuclidean>(point, radius_sq) {
                    if reaches_count(hit.item) {
                        return true;
                    }
                }
            }
            KdTreeBackend::Projected2d { tree, axes } => {
                for hit in
                    tree.within_unsorted::<SquaredEuclidean>(&project2(point, *axes), radius_sq)
                {
                    if squared_distance(&self.xyz_host[hit.item as usize], point) <= radius_sq
                        && reaches_count(hit.item)
                    {
                        return true;
                    }
                }
            }
            KdTreeBackend::Projected1d { tree, axis } => {
                for hit in
                    tree.within_unsorted::<SquaredEuclidean>(&project1(point, *axis), radius_sq)
                {
                    if squared_distance(&self.xyz_host[hit.item as usize], point) <= radius_sq
                        && reaches_count(hit.item)
                    {
                        return true;
                    }
                }
            }
            KdTreeBackend::BruteForce => {
                for (idx, candidate) in self.xyz_host.iter().enumerate() {
                    if squared_distance(candidate, point) <= radius_sq && reaches_count(idx as u64)
                    {
                        return true;
                    }
                }
            }
        };

        false
    }

    pub fn radius_search(&self, query: &[[f32; 3]], radius: f32) -> Result<Vec<Vec<NeighborHit>>> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(PointCloudError::InvalidParameter(
                "radius must be finite and non-negative".to_string(),
            ));
        }
        let radius_sq = radius * radius;
        Ok(query
            .par_iter()
            .map(|point| self.radius_hits(point, radius_sq))
            .collect())
    }

    fn radius_hits(&self, point: &[f32; 3], radius_sq: f32) -> Vec<NeighborHit> {
        let mut hits = self.radius_hits_unsorted(point, radius_sq);
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }

    fn radius_hits_unsorted(&self, point: &[f32; 3], radius_sq: f32) -> Vec<NeighborHit> {
        match &self.backend {
            KdTreeBackend::Kd3d(tree) => tree
                .within_unsorted::<SquaredEuclidean>(point, radius_sq)
                .into_iter()
                .map(|hit| NeighborHit {
                    index: hit.item,
                    distance: hit.distance.sqrt(),
                })
                .collect(),
            KdTreeBackend::LargeKd3d(tree) => tree
                .within_unsorted::<SquaredEuclidean>(point, radius_sq)
                .into_iter()
                .map(|hit| NeighborHit {
                    index: hit.item,
                    distance: hit.distance.sqrt(),
                })
                .collect(),
            KdTreeBackend::Projected2d { tree, axes } => tree
                .within_unsorted::<SquaredEuclidean>(&project2(point, *axes), radius_sq)
                .into_iter()
                .filter_map(|hit| self.full_distance_hit(point, hit.item, radius_sq))
                .collect(),
            KdTreeBackend::Projected1d { tree, axis } => tree
                .within_unsorted::<SquaredEuclidean>(&project1(point, *axis), radius_sq)
                .into_iter()
                .filter_map(|hit| self.full_distance_hit(point, hit.item, radius_sq))
                .collect(),
            KdTreeBackend::BruteForce => brute_force_radius_sq(&self.xyz_host, point, radius_sq),
        }
    }

    fn full_distance_hit(
        &self,
        query: &[f32; 3],
        index: u64,
        radius_sq: f32,
    ) -> Option<NeighborHit> {
        let distance_sq = squared_distance(&self.xyz_host[index as usize], query);
        (distance_sq <= radius_sq).then_some(NeighborHit {
            index,
            distance: distance_sq.sqrt(),
        })
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

#[cfg(test)]
fn brute_force_radius(xyz: &[[f32; 3]], query: &[f32; 3], radius: f32) -> Vec<NeighborHit> {
    brute_force_radius_sq(xyz, query, radius * radius)
}

fn brute_force_radius_sq(xyz: &[[f32; 3]], query: &[f32; 3], radius_sq: f32) -> Vec<NeighborHit> {
    let mut hits: Vec<NeighborHit> = xyz
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| {
            let d_sq = squared_distance(point, query);
            (d_sq <= radius_sq).then_some(NeighborHit {
                index: idx as u64,
                distance: d_sq.sqrt(),
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

fn brute_force_nearest_one(xyz: &[[f32; 3]], query: &[f32; 3]) -> Option<KnnHit> {
    xyz.iter()
        .enumerate()
        .map(|(idx, point)| KnnHit {
            index: idx as u64,
            distance_sq: squared_distance(point, query),
        })
        .min_by(|a, b| {
            a.distance_sq
                .partial_cmp(&b.distance_sq)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    squared_distance(a, b).sqrt()
}

fn squared_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn build_backend(xyz: &[[f32; 3]]) -> KdTreeBackend {
    let constant_axes = constant_axes(xyz);
    match constant_axes.as_slice() {
        [] if !has_axis_bucket_over_limit(xyz, &[0, 1, 2], 32) => {
            let mut tree: KdTree<f32, 3> = KdTree::new();
            for (idx, point) in xyz.iter().enumerate() {
                tree.add(point, idx as u64);
            }
            KdTreeBackend::Kd3d(tree)
        }
        [] if !has_axis_bucket_over_limit(xyz, &[0, 1, 2], PROJECTED_BUCKET_SIZE) => {
            let mut tree: LargeKdTree3d = LargeKdTree3d::new();
            for (idx, point) in xyz.iter().enumerate() {
                tree.add(point, idx as u64);
            }
            KdTreeBackend::LargeKd3d(tree)
        }
        [constant_axis] => {
            let axes = projected_axes(*constant_axis);
            if has_axis_bucket_over_limit(xyz, &axes, PROJECTED_BUCKET_SIZE) {
                log::warn!(
                    "projected point buckets exceed kiddo limits; using brute-force queries"
                );
                KdTreeBackend::BruteForce
            } else {
                let mut tree: ProjectedKdTree2d = ProjectedKdTree2d::new();
                for (idx, point) in xyz.iter().enumerate() {
                    tree.add(&project2(point, axes), idx as u64);
                }
                KdTreeBackend::Projected2d { tree, axes }
            }
        }
        [a, b] => {
            let axis = remaining_axis(*a, *b);
            if has_axis_bucket_over_limit(xyz, &[axis], PROJECTED_BUCKET_SIZE) {
                log::warn!(
                    "projected point buckets exceed kiddo limits; using brute-force queries"
                );
                KdTreeBackend::BruteForce
            } else {
                let mut tree: ProjectedKdTree1d = ProjectedKdTree1d::new();
                for (idx, point) in xyz.iter().enumerate() {
                    tree.add(&project1(point, axis), idx as u64);
                }
                KdTreeBackend::Projected1d { tree, axis }
            }
        }
        _ => {
            log::warn!("degenerate point buckets exceed kiddo limits; using brute-force queries");
            KdTreeBackend::BruteForce
        }
    }
}

fn constant_axes(xyz: &[[f32; 3]]) -> Vec<usize> {
    (0..3)
        .filter(|&axis| xyz.iter().all(|point| point[axis] == xyz[0][axis]))
        .collect()
}

fn has_axis_bucket_over_limit(xyz: &[[f32; 3]], axes: &[usize], limit: usize) -> bool {
    for &axis in axes {
        if axis_bucket_over_limit(xyz, axis, limit) {
            return true;
        }
    }
    false
}

fn axis_bucket_over_limit(xyz: &[[f32; 3]], axis: usize, limit: usize) -> bool {
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
    false
}

fn projected_axes(constant_axis: usize) -> [usize; 2] {
    match constant_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

fn remaining_axis(a: usize, b: usize) -> usize {
    [0, 1, 2]
        .into_iter()
        .find(|axis| *axis != a && *axis != b)
        .unwrap_or(0)
}

fn project2(point: &[f32; 3], axes: [usize; 2]) -> [f32; 2] {
    [point[axes[0]], point[axes[1]]]
}

fn project1(point: &[f32; 3], axis: usize) -> [f32; 1] {
    [point[axis]]
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
    fn duplicate_point_bucket_uses_deterministic_fallback() {
        let xyz: Vec<[f32; 3]> = (0..40).map(|_| [0.0, 0.0, 0.0]).collect();
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();
        assert!(matches!(index.backend, KdTreeBackend::BruteForce));
    }

    #[test]
    fn constant_axis_clouds_use_projected_kdtree() {
        let xyz: Vec<[f32; 3]> = (0..128)
            .map(|i| {
                let x = (i % 16) as f32;
                let y = (i / 16) as f32;
                [x, y, 0.0]
            })
            .collect();
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();
        assert!(matches!(index.backend, KdTreeBackend::Projected2d { .. }));
    }

    #[test]
    fn constant_two_axis_clouds_use_projected_kdtree() {
        let xyz: Vec<[f32; 3]> = (0..40).map(|i| [0.0, i as f32, 0.0]).collect();
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();
        assert!(matches!(index.backend, KdTreeBackend::Projected1d { .. }));

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

    #[test]
    fn kdtree_index_is_sync_for_parallel_queries() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KdTreeIndex>();
    }

    #[test]
    fn self_radius_counts_exclude_only_matching_item_id() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        .unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();

        assert_eq!(
            index.radius_self_has_at_least(0.01, 1).unwrap(),
            vec![true, true, false]
        );
    }

    #[test]
    fn self_knn_mean_distances_are_euclidean_and_exclude_self_by_id() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [6.0, 8.0, 0.0],
        ])
        .unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();

        assert_eq!(
            index.knn_mean_distances_for_index_points(1).unwrap(),
            vec![5.0, 5.0, 5.0]
        );
    }

    #[test]
    fn knn_one_within_returns_squared_distance() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
            .unwrap();
        let index = KdTreeIndex::build(&pc).unwrap();

        let hits = index.knn_one_within(&[[3.0, 0.0, 0.0]], 10.0).unwrap();
        let hit = hits[0].unwrap();
        assert_eq!(hit.index, 0);
        assert_eq!(hit.distance_sq, 9.0);
    }

    fn hit_ids(rows: &[Vec<NeighborHit>]) -> Vec<Vec<u64>> {
        rows.iter()
            .map(|row| row.iter().map(|hit| hit.index).collect())
            .collect()
    }
}
