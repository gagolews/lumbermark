# distutils: language=c++
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
# cython: language_level=3


"""
Internal functions and classes
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


cimport libc.math
from libcpp cimport bool

import numpy as np
cimport numpy as np
np.import_array()

# import warnings
#import os

#from libcpp.vector cimport vector
#from libc.math cimport INFINITY
#from libcpp.vector cimport vector


ctypedef fused T:
    int
    long
    long long
    Py_ssize_t
    float
    double


ctypedef fused floatT:
    float
    double


cdef extern from "c_lumbermark.h":
    cdef cppclass CLumbermark[T]:

        CLumbermark() except +

        CLumbermark(
            T* mst_d, Py_ssize_t* mst_i, Py_ssize_t m, Py_ssize_t n,
            bool skip_leaves, const Py_ssize_t* cumdeg, const Py_ssize_t* inc
        ) except +

        Py_ssize_t compute(
            Py_ssize_t n_clusters, Py_ssize_t min_cluster_size,
            T min_cluster_factor
        ) except +

        void get_labels(Py_ssize_t* res)
        void get_cut_edges(Py_ssize_t* res)
        # void get_is_unreachable(bint* res)


cpdef dict lumbermark_from_mst(
        floatT[::1] mst_d,
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t[::1] mst_cumdeg,
        Py_ssize_t[::1] mst_inc,
        Py_ssize_t n_clusters,
        Py_ssize_t min_cluster_size=10,
        floatT min_cluster_factor=0.15,
        bool skip_leaves=True
    ):
    """
    lumbermark.lumbermark_from_mst(
        mst_d, mst_i, mst_cumdeg, mst_inc,
        n_clusters, min_cluster_size=10, min_cluster_factor=0.15, skip_leaves=True
    )

    The Lumbermark Clustering Algorithm

    Determines a dataset's partition based on a precomputed spanning tree.


    Parameters
    ----------

    mst_d, mst_i : ndarray
        a spanning tree with `m=n-1` edges defined by a pair
        `(mst_i, mst_d)`; see ``quitefastmst.mst_euclid``

    mst_cumdeg : ndarray, length n+1
        see `deadwood.graph_vertex_incidences`

    mst_inc : ndarray, length 2*m
        see `deadwood.graph_vertex_incidences`

    n_clusters : int
        the number of clusters requested

    min_cluster_size : int
        minimal cluster size

    min_cluster_factor : float
        output cluster sizes won't be smaller than
        `min_cluster_factor/n_clusters*n_points` (excluding outliers)

    skip_leaves : bool
        whether the MST leaves should be omitted from cluster size counting


    Returns
    -------

    res : dict, with the following elements:
        n_clusters : integer
            actual number of clusters found, 0 if ``labels`` is ``None``

        labels : ndarray, shape (n,)
            ``labels[i]`` gives the cluster ID of the `i`-th input point;
            a number between `0` and `n_clusters-1`

        cut_edges : ndarray, shape (n_clusters-1, )
            indexes of the cut edges of the spanning tree; their removal
            leads to the formation of clusters (connected components)


    References
    ----------

    .. [1]
        M. Gagolewski, Lumbermark: Resistant clustering by chopping up mutual
        reachability minimum spanning trees, 2026,
        https://doi.org/10.48550/arXiv.2604.07143
    """
    cdef Py_ssize_t m = mst_i.shape[0]
    cdef Py_ssize_t n = m+1

    if not m == mst_d.shape[0] or m != n-1 or mst_i.shape[1] != 2:
        raise ValueError("ill-defined spanning tree")

    if mst_cumdeg.shape[0] != n+1:
        raise ValueError("mst_cumdeg should be of length n+1")

    if mst_inc.shape[0] != 2*m:
        raise ValueError("mst_inc should be of length 2*m")

    if not 1 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")

    cdef np.ndarray[Py_ssize_t] labels_
    cdef np.ndarray[Py_ssize_t] cut_edges_
    cdef np.ndarray[bool] is_unreachable_
    cdef Py_ssize_t n_clusters_

    cdef CLumbermark[floatT] lm
    lm = CLumbermark[floatT](
        &mst_d[0], &mst_i[0,0], m, n, skip_leaves, &mst_cumdeg[0], &mst_inc[0]
    )

    n_clusters_ = lm.compute(
        n_clusters, min_cluster_size, min_cluster_factor
    )

    if n_clusters_ <= 0:
        raise RuntimeError("no clusters detected")
    # elif n_clusters_ != n_clusters:
    #     warnings.warn("the number of clusters detected does not match the requested one")

    cut_edges_ = np.empty(n_clusters_-1, dtype=np.intp)
    lm.get_cut_edges(&cut_edges_[0])

    labels_ = np.empty(n, dtype=np.intp)
    lm.get_labels(&labels_[0])

    return dict(
        n_clusters=n_clusters_,
        labels=labels_,
        cut_edges=cut_edges_,
    )
