# Registration

The `pcl_rustic.registration` module provides ICP-family registration.

## Point-To-Point ICP

```python
import numpy as np
from pcl_rustic import registration

result = registration.icp(
    source,
    target,
    max_correspondence_distance=0.05,
    init=np.eye(4, dtype=np.float32),
    estimation=registration.TransformationEstimation.point_to_point(),
    criteria=registration.ICPConvergenceCriteria(
        max_iteration=30,
        relative_fitness=1e-6,
        relative_rmse=1e-6,
    ),
)

print(result.transformation)
print(result.fitness, result.inlier_rmse)
```

## Point-To-Plane ICP

Point-to-plane ICP requires target normals:

```python
from pcl_rustic import NormalSearch, registration

target.estimate_normals(NormalSearch.knn(30))
result = registration.icp(
    source,
    target,
    0.05,
    np.eye(4, dtype=np.float32),
    registration.TransformationEstimation.point_to_plane(),
)
```

## Generalized ICP

GICP requires packed covariance attributes on both clouds:

```python
source.estimate_covariances(knn=30)
target.estimate_covariances(knn=30)

result = registration.icp(
    source,
    target,
    0.05,
    np.eye(4, dtype=np.float32),
    registration.TransformationEstimation.generalized(epsilon=1e-3),
)
```

The current GICP path validates and carries covariance data; the iterative update
uses the stable point-to-point solve as the baseline estimator.

## Evaluation

```python
score = registration.evaluate(
    source,
    target,
    max_correspondence_distance=0.05,
    transformation=np.eye(4, dtype=np.float32),
)
```

`RegistrationResult.correspondence_set` contains `(source_index, target_index)`
pairs.

