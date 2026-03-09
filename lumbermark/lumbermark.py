"""
The Lumbermark Clustering Algorithm
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2025-2026, Marek Gagolewski <https://www.gagolewski.com>      #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


import os
import sys
import math
import numpy as np
from . import core
import warnings
import deadwood


###############################################################################
###############################################################################
###############################################################################


class Lumbermark(deadwood.MSTClusterer):
    """
    Lumbermark [1]_ is a fast and robust divisive clustering algorithm
    which identifies a specified number of clusters.

    It iteratively chops off sizeable limbs that are joined by protruding
    segments of a dataset's mutual reachability minimum spanning tree.

    The use of a mutual reachability distance [2]_ pulls peripheral points
    farther away from each other.  When combined with the ``deadwood`` package,
    it can act as an outlier detector.

    Once the spanning tree is determined (:math:`\\Omega(n \\log n)` –
    :math:`O(n^2)`), the Lumbermark algorithm runs in :math:`O(kn)` time,
    where :math:`k` is the number of clusters sought.  Memory use is
    :math:`O(n)`.

    As with all distance-based methods (this includes k-means and DBSCAN as
    well), applying data preprocessing and feature engineering techniques
    (e.g., feature scaling, feature selection, dimensionality reduction)
    might lead to more meaningful results.


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect.

    min_cluster_size : int, default=10
        Minimal cluster size.

    min_cluster_factor : float in [0,1], default=0.25
        Output cluster sizes will not be smaller than
        ``min_cluster_factor*n_points/n_clusters``.

    skip_leaves : bool, default='auto'
        Whether the MST leaves should be omitted from cluster size counting;
        ``"auto"`` selects True if `M > 0`.

    M : int, default=5
        Smoothing factor for the mutual reachability distance [2]_.
        `M = 0` and `M = 1` select the original distance as given by
        the `metric` parameter; see :any:`deadwood.MSTBase`.

    metric : str, default='l2'
        The metric used to compute the linkage; see :any:`deadwood.MSTBase`
        for more details.  Defaults to the Euclidean distance.

    quitefastmst_params : dict
        Additional parameters to be passed to ``quitefastmst.mst_euclid``
        if ``metric`` is ``"l2"``.

    verbose : bool, default=False
        Whether to print diagnostic messages and progress information
        onto ``stderr``.


    Attributes
    ----------

    labels_ : ndarray
        Detected cluster labels.

        It is an integer vector such that ``labels_[i]`` gives
        the cluster ID (between 0 and `n_clusters_` - 1) of the `i`-th object.

    n_clusters_ : int
        The actual number of clusters detected by the algorithm.

    n_samples_ : int
        The number of points in the dataset.

    n_features_ : int or None
        The number of features in the dataset.


    References
    ----------

    .. [1]
        M. Gagolewski, *Lumbermark*, in preparation, 2026, TODO

    .. [2]
        R.J.G.B. Campello, D. Moulavi, J. Sander,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        https://doi.org/10.1007/978-3-642-37456-2_14

    .. [3]
        M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        https://doi.org/10.1007/s00357-024-09483-1
    """
    def __init__(
            self,
            n_clusters=2,
            *,
            min_cluster_size=10,
            min_cluster_factor=0.25,
            skip_leaves="auto",
            M=5,
            metric="l2",
            quitefastmst_params=dict(),  # dist_min is generally better than dcore_min
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__(
            n_clusters=n_clusters,
            M=M,
            metric=metric,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.min_cluster_size      = min_cluster_size
        self.min_cluster_factor    = min_cluster_factor
        self.skip_leaves           = skip_leaves


    def _check_params(self):
        super()._check_params()

        self.min_cluster_factor = float(self.min_cluster_factor)
        if not (0.0 <= self.min_cluster_factor <= 1.0):
            raise ValueError("min_cluster_factor must be in [0,1].")

        self.min_cluster_size = int(self.min_cluster_size)
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1.")

        if self.skip_leaves != "auto":
            self.skip_leaves = bool(self.skip_leaves)


    def fit(self, X, y=None):
        """
        Performs a cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix or a data frame with ``n_samples`` rows
            and ``n_features`` columns;
            see :any:`deadwood.MSTBase.fit_predict` for more details.

        y : None
            Ignored.


        Returns
        -------

        self : lumbermark.Lumbermark
            The object that the method was called on.


        Notes
        -----

        Refer to the `labels_` and `n_clusters_` attributes for the result.
        """
        self.labels_     = None
        self.n_clusters_ = None
        self._cut_edges_ = None

        self._check_params()  # re-check, they might have changed
        self._get_mst(X)  # sets n_samples_, n_features_, _tree_d, _tree_i, _d_core, etc.

        if not (1 <= self.n_clusters < self.n_samples_):
            raise ValueError("n_clusters must be between 1 and n_samples_-1")

        if self.verbose:
            print("[lumbermark] Determining clusters with Lumbermark.", file=sys.stderr)

        # apply the Lumbermark algorithm:
        res = core.lumbermark_from_mst(
            self._tree_d_,
            self._tree_i_,
            self._tree_cumdeg_,
            self._tree_inc_,
            n_clusters=self.n_clusters,
            min_cluster_size=self.min_cluster_size,
            min_cluster_factor=self.min_cluster_factor,
            skip_leaves=self.skip_leaves if self.skip_leaves!="auto" else (self.M>0)
        )

        self.n_clusters_ = res["n_clusters"]
        self.labels_     = res["labels"]
        self._cut_edges_ = res["cut_edges"]
        #self._iters_     = res["iters"]

        if self.n_clusters_ != self.n_clusters:
            warnings.warn("The number of clusters detected (%d) is "
                          "different from the requested one (%d)." % (
                            self.n_clusters_,
                            self.n_clusters))

        if self.verbose:
            print("[lumbermark] Done.", file=sys.stderr)

        return self
