"""
The Lumbermark Clustering Algorithm
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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
from . import internal
import warnings
import genieclust

###############################################################################
###############################################################################
###############################################################################


class Lumbermark(genieclust.MSTClusterMixin):
    """
    Lumbermark: TODO DESCRIBE


    Parameters
    ----------

    n_clusters : int
        The number of clusters to detect.

        TODO If *M > 1* and *postprocess* is not ``"all"``, setting
        *n_clusters = 1* makes the algorithm behave like a noise
        point/outlier detector.

    min_cluster_size : int
        Minimal cluster size.

    min_cluster_factor : float in [0,1]
        Output cluster sizes will not be smaller than
        ``min_cluster_factor*n_points/n_clusters``, where
        *n_points* excludes noise and boundary points (TODO: confirm)

    M : int
        The smoothing factor for the mutual reachability distance [2]_.
        *M = 1* and *M = 2* indicate the original distance as given by
        the *metric* parameter; see :any:`genieclust.MSTClusterMixin`
        for more details.

    metric : str
        The metric used to compute the linkage; see
        :any:`genieclust.MSTClusterMixin` for more details.
        Defaults to ``"l2"``, i.e., the Euclidean distance.

    preprocess : TODO
        TODO

    postprocess : {``"none"``, ``"all"``}
        Controls the treatment of noise/boundary points once the clusters are
        identified.

        ``"none"`` leaves noise points as-is.
        ``"all"`` merges them with the nearest clusters.

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
        If *M > 1*, noise/boundary points are labelled ``-1`` (unless taken care
        of at the postprocessing stage).

    n_clusters_ : int
        The actual number of clusters detected by the algorithm.

    n_samples_ : int
        The number of points in the dataset.

    n_features_ : int
        The number of features in the dataset.

        If the information is not available, it will be set to ``-1``.


    Notes
    -----

    TODO: describe Lumbermark
    A robust divisive clustering algorithm based on spanning trees,
    aiming to detect a specific number of clusters.


    The Genie algorithm itself has :math:`O(TODO)` time complexity provided
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

    Note that *M = 2* corresponds to the original distance.
    If *M > 1*, all MST leaves are left out from the clustering process.
    They may be merged with the nearest clusters at the postprocessing stage,
    or left marked as "noise" observations.


    :Environment variables:
        OMP_NUM_THREADS
            Controls the number of threads used when computing the minimum
            spanning tree.


    References
    ----------

    .. [1]
        TODO

    .. [2]
        Campello, R.J.G.B., Moulavi, D., Sander, J.,
        Density-based clustering based on hierarchical density estimates,
        *Lecture Notes in Computer Science* 7819, 2013, 160-172,
        doi:10.1007/978-3-642-37456-2_14.

    .. [3]
        Gagolewski, M., Cena, A., Bartoszuk, M., Brzozowski, L.,
        Clustering with minimum spanning trees: How good can it be?,
        *Journal of Classification* 42, 2025, 90-112,
        doi:10.1007/s00357-024-09483-1.

    """
    def __init__(
            self,
            *,
            n_clusters=2,
            min_cluster_size=10,
            min_cluster_factor=0.15,
            M=0,
            metric="l2",
            preprocess="auto",  # TODO
            postprocess="none", # TODO
            quitefastmst_params=dict(mutreach_ties="dcore_min", mutreach_leaves="reconnect_dcore_min"),
            verbose=False
        ):
        # # # # # # # # # # # #
        super().__init__(
            n_clusters=n_clusters,
            M=M,
            metric=metric,
            #preprocess=preprocess,
            #postprocess=postprocess,
            quitefastmst_params=quitefastmst_params,
            verbose=verbose
        )

        self.min_cluster_size      = min_cluster_size
        self.min_cluster_factor    = min_cluster_factor
        self.preprocess            = preprocess
        self.postprocess           = postprocess
        self._check_params()


    def _check_params(self, cur_state=None):
        cur_state = super()._check_params(cur_state)

        cur_state["min_cluster_factor"] = float(self.min_cluster_factor)
        if not (0.0 <= cur_state["min_cluster_factor"] <= 1.0):
            raise ValueError("`min_cluster_factor` not in [0,1].")

        cur_state["min_cluster_size"] = int(self.min_cluster_size)
        if cur_state["min_cluster_size"] < 1:
            raise ValueError("`min_cluster_size` must be >= 1.")

        _preprocess_options = ("auto", "none", "leaves")  # TODO
        cur_state["preprocess"] = str(self.preprocess).lower()
        if cur_state["preprocess"] not in _preprocess_options:
            raise ValueError("`preprocess` should be one of %s" % repr(_preprocess_options))

        _postprocess_options = ("none", "all")  # TODO
        cur_state["postprocess"] = str(self.postprocess).lower()
        if cur_state["postprocess"] not in _postprocess_options:
            raise ValueError("`postprocess` should be one of %s" % repr(_postprocess_options))

        return cur_state


    def fit(self, X, y=None):
        """
        Perform cluster analysis of a dataset.


        Parameters
        ----------

        X : object
            Typically a matrix with ``n_samples`` rows and ``n_features``
            columns; see :any:`genieclust.MSTClusterMixin.fit_predict` for more
            details.

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
        cur_state = self._check_params()  # re-check, they might have changed

        cur_state = self._get_mst(X, cur_state)

        if cur_state["n_clusters"] >= self.n_samples_:
            raise ValueError("n_clusters must be < n_samples_")

        if cur_state["preprocess"] == "auto":
            if cur_state["M"] > 0:
                cur_state["preprocess"] = "leaves"
            else:
                cur_state["preprocess"] = "none"

        if cur_state["verbose"]:
            print("[lumbermark] Determining clusters with Lumbermark.", file=sys.stderr)

        # apply the Lumbermark algorithm:
        res = internal.lumbermark_from_mst(
            self._tree_w,
            self._tree_e,
            n_clusters=cur_state["n_clusters"],
            min_cluster_size=cur_state["min_cluster_size"],
            min_cluster_factor=cur_state["min_cluster_factor"],
            skip_leaves=(cur_state["preprocess"] == "leaves")
        )

        self.labels_     = res["labels"]
        self._links_     = res["links"]
        self._iters_     = res["iters"]

        if res["n_clusters"] != cur_state["n_clusters"]:
            warnings.warn("The number of clusters detected (%d) is "
                          "different from the requested one (%d)." % (
                            res["n_clusters"],
                            cur_state["n_clusters"]))
        self.n_clusters_ = res["n_clusters"]

        if cur_state["postprocess"] == "none" and cur_state["preprocess"] == "leaves":
            res["labels"][res["is_noise"]] = -1

        # TODO: postprocess midliers????

        if cur_state["verbose"]:
            print("[lumbermark] Done.", file=sys.stderr)

        return self

