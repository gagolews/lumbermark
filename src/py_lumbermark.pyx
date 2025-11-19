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


cimport libc.math
from libcpp cimport bool

import numpy as np
cimport numpy as np
np.import_array()

import warnings
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



cdef extern from "../src/c_lumbermark.h":
    cdef cppclass CLumbermark[T]:
        CLumbermark() except +
        CLumbermark(T* mst_d, Py_ssize_t* mst_i, Py_ssize_t m, Py_ssize_t n) except +
        Py_ssize_t compute(
            Py_ssize_t n_clusters, Py_ssize_t min_cluster_size,
            T min_cluster_factor
        ) except +
        void get_labels(Py_ssize_t* res)
        void get_links(Py_ssize_t* res)
        void get_is_unreachable(bint* res)



cpdef dict lumbermark_from_mst(
        floatT[::1] mst_d,
        Py_ssize_t[:,::1] mst_i,
        Py_ssize_t n,
        Py_ssize_t n_clusters,
        Py_ssize_t min_cluster_size=10,
        floatT min_cluster_factor=0.15
    ):
    """The Lumbermark Clustering Algorithm

    Determines a dataset's partition based on a precomputed MST.

    TODO: citation
    Gagolewski, M., TODO, 2025



    Parameters
    ----------

    mst_d, mst_i : ndarray
        A spanning tree defined by a pair (mst_i, mst_d);
        see genieclust.mst.  Actually, not all points must be reachable;
        in such a case, they are treated as outliers.
    n : int
        Number of points in the dataset.
    n_clusters : int
        Number of clusters the dataset is split into.
    min_cluster_size : int
        Minimal cluster size.
    min_cluster_factor : float
        Output cluster sizes won't be smaller than
        min_cluster_factor/n_clusters*n_points (excluding outliers)


    Returns
    -------

    res : dict, with the following elements:
        labels : ndarray, shape (n,)

            labels[i] gives the cluster id of the i-th input point;
            a number between 0 and n_clusters-1.

        n_clusters : integer
            actual number of clusters found, 0 if labels is None.

        iters : None
            unused.

        links : ndarray, shape (n_clusters-1, )
            cut edges (indexes) of the spanning tree whose removal
            leads to the formation of clusters (connected components).

        is_unreachable : ndarray, shape (n,)
            is_unreachable[i] is True if the i-th point is unreachable.


    """
    cdef Py_ssize_t m = mst_i.shape[0]

    if not m == mst_d.shape[0] or m > n-1:
        raise ValueError("ill-defined MST")

    if not 1 <= n_clusters <= n:
        raise ValueError("incorrect n_clusters")

    cdef np.ndarray[Py_ssize_t] labels_
    cdef np.ndarray[Py_ssize_t] links_
    cdef np.ndarray[bool] is_unreachable_
    cdef Py_ssize_t n_clusters_

    cdef CLumbermark[floatT] l
    l = CLumbermark[floatT](&mst_d[0], &mst_i[0,0], m, n)

    n_clusters_ = l.compute(
        n_clusters, min_cluster_size, min_cluster_factor
    )

    if n_clusters_ <= 0:
        raise RuntimeError("no clusters detected")
    elif n_clusters_ != n_clusters:
        warnings.warn("the number of clusters detected does not match the requested one")

    links_ = np.empty(n_clusters_-1, dtype=np.intp)
    l.get_links(&links_[0])

    labels_ = np.empty(n, dtype=np.intp)
    l.get_labels(&labels_[0])

    is_unreachable_ = np.empty(n, dtype=np.bool)
    l.get_is_unreachable(&is_unreachable_[0])

    return dict(
        labels=labels_,
        n_clusters=n_clusters_,
        iters=None,
        links=links_,
        is_unreachable=is_unreachable_
    )
