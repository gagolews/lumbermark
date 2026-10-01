# Changelog

## TODO

*   A multi-threaded version of Lumbermark.
*   A way to assign new points to existing clusters (predict)
    based on the labels of points' nearest-neighbours.


## 0.9.1 (2026-10-01)

*   [Python, R] [BACKWARD INCOMPATIBILITY]  Lumbermark is now equipped with
    the `nested` parameter, which defaults to `True`, resulting in the same
    partition hierarchy regardless of `n_clusters` selected.  For the previous
    behaviour (a quasi-hierarchical algorithm which might be slightly faster),
    select `nested=False`.


## 0.9.0 (2026-03-09)

*   [Python] Initial PyPI release.

*   [R] Initial CRAN release.
