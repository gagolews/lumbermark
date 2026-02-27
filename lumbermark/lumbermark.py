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
    Lumbermark: TODO DESCRIBE


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect.

    min_cluster_size : int
        Minimal cluster size.

    min_cluster_factor : float in [0,1]
        Output cluster sizes will not be smaller than
        ``min_cluster_factor*n_points/n_clusters``

    M : int
        Smoothing factor for the mutual reachability distance [2]_.
        `M = 0` and `M = 1` select the original distance as given by
        the `metric` parameter; see :any:`deadwood.MSTBase`

    metric : str, default='l2'
        The metric used to compute the linkage; see :any:`deadwood.MSTBase`
        for more details.  Defaults to the Euclidean distance.

    quitefastmst_params : dict
        Additional parameters to be passed to ``quitefastmst.mst_euclid``
        if ``metric`` is ``"l2"``

    verbose : bool
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


    Notes
    -----

    TODO: describe Lumbermark
    A robust divisive clustering algorithm based on spanning trees,
    aiming to detect a specific number of clusters.


    The Deadwood algorithm itself has :math:`O(TODO)` time complexity provided
    that a minimum spanning tree of the pairwise distance graph is given.
    If the Euclidean distance is selected, then
    ``quitefastmst.mst_euclid`` is used to compute the MST;
    it is quite fast in low-dimensional spaces.
    Otherwise, an implementation of the Jarník (Prim/Dijkstra)-like
    :math:`O(n^2)`-time algorithm is called.

    The Lumbermark algorithm with the smoothing factor *M > 1*
    controlling the mutual reachability distance gives a version of the
    HDBSCAN\\* [2]_ algorithm that, contrary to its predecessor, is able to
    detect a *predefined* number of clusters.

    Note that *M = 1* corresponds to the original distance.


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used when computing the minimum
            spanning tree.


    References
    ----------

    .. [1]
        M. Gagolewski, *Lumbermark*, in preparation, 2026, TODO

    .. [2]
        R.J.G.B. Campello, D. Moulavi, J. Sander,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        https://doi.org/10.1007/978-3-642-37456-2_14

    .. [3]
        M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        https://doi.org/10.1007/s00357-024-09483-1
    """
    def __init__(
            self,
            n_clusters=2,
            *,
            min_cluster_size=10,
            min_cluster_factor=0.20,
            M=5,  # TODO set default
            metric="l2",
            quitefastmst_params=dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
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


    def _check_params(self):
        super()._check_params()

        self.min_cluster_factor = float(self.min_cluster_factor)
        if not (0.0 <= self.min_cluster_factor <= 1.0):
            raise ValueError("min_cluster_factor must be in [0,1].")

        self.min_cluster_size = int(self.min_cluster_size)
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be >= 1.")


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
            min_cluster_factor=self.min_cluster_factor
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

