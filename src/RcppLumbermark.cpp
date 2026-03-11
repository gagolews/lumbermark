/*  Lumbermark R interface
 *
 *  Copyleft (C) 2025-2026, Marek Gagolewski <https://www.gagolewski.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If this is not the case, refer to <https://www.gnu.org/licenses/>.
 */


#include "c_common.h"
#include "c_lumbermark.h"

using namespace Rcpp;


/*! Compute the degree of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *  NOTE This was taken from 'deadwood'!
 *  TODO export Cgraph_vertex_degrees in deadwood
 *
 * @param ind c_contiguous matrix of size m*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n
 * @param m number of edges (rows in ind)
 * @param n number of vertices
 * @param deg [out] array of size n, where
 *     deg[i] will give the degree of the i-th vertex.
 */
void Cgraph_vertex_degrees(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    Py_ssize_t* deg /*out*/
) {
    for (Py_ssize_t i=0; i<n; ++i)
        deg[i] = 0;

    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];

        if (u < 0 || v < 0)
            throw std::domain_error("All elements must be >= 0");
        else if (u >= n || v >= n)
            throw std::domain_error("All elements must be < n");
        else if (u == v)
            throw std::domain_error("Self-loops are not allowed");

        deg[u]++;
        deg[v]++;
    }
}


/*! Compute the incidence list of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *  NOTE This was taken from 'deadwood'!
 *  TODO export Cgraph_vertex_incidences in deadwood
 *
 *  @param ind c_contiguous matrix of size m*2,
 *      where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n
 *  @param m number of edges (rows in ind)
 *  @param n number of vertices
 *  @param cumdeg [out] array of size n+1, where cumdeg[i+1] the sum of the first i vertex degrees
 *  @param inc [out] array of size 2*m; inc[cumdeg[i]]..inc[cumdeg[i+1]-1] gives the edges incident on the i-th vertex
 */
void Cgraph_vertex_incidences(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    Py_ssize_t* cumdeg,
    Py_ssize_t* inc
) {
    cumdeg[0] = 0;
    Cgraph_vertex_degrees(ind, m, n, cumdeg+1);

    Py_ssize_t cd = 0;
    for (Py_ssize_t i=1; i<n+1; ++i) {
        Py_ssize_t this_deg = cumdeg[i];
        cumdeg[i] = cd;
        cd += this_deg;
    }
    // that's not it yet; cumdeg is adjusted below


    for (Py_ssize_t e=0; e<m; ++e) {
        Py_ssize_t u = ind[2*e+0];
        Py_ssize_t v = ind[2*e+1];

        *(inc+cumdeg[u+1]) = e;
        ++(cumdeg[u+1]);

        *(inc+cumdeg[v+1]) = e;
        ++(cumdeg[v+1]);
    }

    LUMBERMARK_ASSERT(cumdeg[0] == 0);
    LUMBERMARK_ASSERT(cumdeg[n] == 2*m);
}

// [[Rcpp::export(".lumbermark")]]
IntegerVector dot_lumbermark(
    NumericMatrix mst,
    int k,
    int min_cluster_size,
    double min_cluster_factor,
    bool skip_leaves,
    bool verbose
)
{
    if (verbose) LUMBERMARK_PRINT("[lumbermark] Determining clusters.\n");

    if (min_cluster_factor <= 0.0 || min_cluster_factor >= 1.0)
        stop("`min_cluster_factor` must be in (0, 1)");

    Py_ssize_t n = mst.nrow()+1;

    if (k < 1 || k > n) stop("invalid number of clusters requested, `k`");

    std::vector<Py_ssize_t> mst_i((n-1)*2);
    std::vector<double> mst_d(n-1);
    for (Py_ssize_t i=0; i<n-1; ++i) {
        mst_i[2*i+0] = (Py_ssize_t)mst(i, 0) - 1;  // 1-based to 0-based indexes, to C order
        mst_i[2*i+1] = (Py_ssize_t)mst(i, 1) - 1;
        mst_d[i] = mst(i, 2);
    }

    std::vector<Py_ssize_t> cumdeg(n+1);
    std::vector<Py_ssize_t> inc(2*(n-1));
    Cgraph_vertex_incidences(mst_i.data(), n-1, n, cumdeg.data(), inc.data());

    CLumbermark lm(mst_d.data(), mst_i.data(), n-1, n, skip_leaves, cumdeg.data(), inc.data());

    int k_detected = lm.compute(
        k, min_cluster_size, min_cluster_factor
    );

    LUMBERMARK_ASSERT(k_detected>0);

    std::vector<Py_ssize_t> res_(n);
    lm.get_labels(res_.data());

    IntegerVector res(n);
    for (Py_ssize_t i=0; i<n; ++i) {
        LUMBERMARK_ASSERT(res_[i] >= 0);
        if (res_[i] < 0) res[i] = NA_INTEGER;  // outlier/noise point TODO: remove
        else res[i] = res_[i] + 1;  // 1-based indexes
    }

    std::vector<Py_ssize_t> cut_edges_(k_detected-1);
    lm.get_cut_edges(cut_edges_.data());
    NumericVector cut_edges(k_detected-1);
    for (Py_ssize_t i=0; i<k_detected-1; ++i) {
        cut_edges[i] = cut_edges_[i]+1;  // 1-based
    }
    res.attr("cut_edges") = cut_edges;

    if (verbose) LUMBERMARK_PRINT("[lumbermark] Done.\n");

    return res;
}
