/*  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
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


#ifndef __c_lumbermark_h
#define __c_lumbermark_h

#include "c_common.h"
#include <algorithm>
#include <vector>
#include <limits>

#define LUMBERMARK_UNSET     (std::numeric_limits<Py_ssize_t>::max())


/*!  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
 *   Splits a spanning tree into a given number of
 *   connected components.
 *
 *
 *   References
 *   ==========
 *
 *   M. Gagolewski, *Lumbermark*, in preparation, 2026, TODO
 */
template <class FLOAT>
class CLumbermark {
protected:

    const FLOAT* mst_d;       //<! m edge weights, sorted increasingly
    const Py_ssize_t* mst_i;  //<! 2*m edge definitions
    Py_ssize_t m;             //<! number of edges, must be n-1
    Py_ssize_t n;             //<! number of points
    bool skip_leaves;         //<! whether the MST leaves should be omitted from cluster counting

    const Py_ssize_t* cumdeg;  // length n+1; see Cgraph_vertex_incidences in 'deadwood'
    const Py_ssize_t* inc;     // length 2*m; see Cgraph_vertex_incidences in 'deadwood'


    // auxiliary data for generating clustering results:
    std::vector<Py_ssize_t> labels;        //<! node labels, size n, in 1..n_clusters and -1..-n_clusters (outliers/noise points)
    std::vector<Py_ssize_t> mst_labels;    //<! edge labels, size m
    std::vector<Py_ssize_t> mst_cutsizes;  //<! size m, each pair gives the sizes of the clusters that are formed when we cut out the corresponding edge

    std::vector<Py_ssize_t> cluster_sizes; //<! size n_clusters
    std::vector<Py_ssize_t> cut_edges;     //<! size n_clusters-1


    /*! vertex visitor:
     *  going from v, visits w and then all its neighbours, mst_i[e,:] = {v,w};
     *  marks them as members of v's cluster and determines the size of its
     *  connected component */
    Py_ssize_t visit(Py_ssize_t v, Py_ssize_t e)
    {
        if (mst_labels[e] < 0)
            return 0;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];
        // so v == mst_i[2*e+iv]

        mst_labels[e] = labels[w] = labels[v];

        Py_ssize_t tot = (skip_leaves && is_leaf(v));  // how many vertices in total?
        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) tot += visit(w, *pe);
        }
        mst_cutsizes[e] = tot;

        return tot;
    }


    void init_labels()
    {
        for (Py_ssize_t v=0; v<n; ++v) labels[v]       = LUMBERMARK_UNSET;
        for (Py_ssize_t e=0; e<m; ++e) mst_labels[e]   = LUMBERMARK_UNSET;
        for (Py_ssize_t e=0; e<m; ++e) mst_cutsizes[e] = LUMBERMARK_UNSET;

        // visit all nodes starting from a vertex incident to edge 0
        Py_ssize_t v = mst_i[2*0+0];
        labels[v] = 0;

        Py_ssize_t tot = (skip_leaves && is_leaf(v));
        for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
            tot += visit(v, *pe);
        }
        cluster_sizes[labels[v]] = tot;

        // ensure all vertices and edges are reachable:
        for (Py_ssize_t v=0; v<n; ++v)
            LUMBERMARK_ASSERT(labels[v] != LUMBERMARK_UNSET);
        for (Py_ssize_t e=0; e<m; ++e)
            LUMBERMARK_ASSERT(mst_labels[e] != LUMBERMARK_UNSET);
    }


public:
    CLumbermark() : CLumbermark(nullptr, nullptr, 0, 0, false, nullptr, nullptr) { }

    CLumbermark(
        const FLOAT* mst_d, const Py_ssize_t* mst_i, Py_ssize_t m, Py_ssize_t n,
        bool skip_leaves, const Py_ssize_t* cumdeg, const Py_ssize_t* inc
    ) :
        mst_d(mst_d), mst_i(mst_i), m(m), n(n), skip_leaves(skip_leaves),
        cumdeg(cumdeg), inc(inc), labels(n), mst_labels(m), mst_cutsizes(m)
    {
        if (n == 0) return;

        if (m != n-1)
            throw std::domain_error("m != n-1");

        for (Py_ssize_t e=1; e<m; ++e) {
            // check if edge weights are sorted increasingly
            LUMBERMARK_ASSERT(mst_d[e-1] <= mst_d[e]);
        }
    }


    inline bool is_leaf(Py_ssize_t v) const {
        return (cumdeg[v+1]-cumdeg[v] <= 1);
    }


    /*! Run the Lumbermark algorithm
     *
     * @param n_clusters number of clusters to find
     * @param min_cluster_size minimal cluster size
     * @param min_cluster_factor output cluster sizes won't be smaller than
     *    min_cluster_factor*n_points/n_clusters

     *
     * @return number of clusters detected (can be smaller than the requested one)
     */
    Py_ssize_t compute(
        Py_ssize_t n_clusters,
        Py_ssize_t min_cluster_size,
        FLOAT min_cluster_factor
    ) {
        LUMBERMARK_ASSERT(n > 2);

        if (n_clusters < 1)
            throw std::domain_error("n_clusters must be >= 1");

        Py_ssize_t n_skip = 0;  // count unreachable vertices
        if (skip_leaves) {
            for (Py_ssize_t v=0; v<n; ++v)
                if (is_leaf(v)) n_skip++;
        }

        min_cluster_size = std::max(
            min_cluster_size,
            (Py_ssize_t)(min_cluster_factor*(n-n_skip)/(FLOAT)n_clusters)
        );

        cut_edges.resize(n_clusters-1);
        cluster_sizes.resize(n_clusters);

        init_labels();

        Py_ssize_t n_clusters_ = 1;
        Py_ssize_t e_last = m;  // edges are consumed in decreasing order

        while (n_clusters_ < n_clusters)
        {
            // find the longest unconsumed edge to cut out
            do {
                e_last--;
                if (e_last < 0) {
                    cut_edges.resize(n_clusters_-1);
                    return n_clusters_;  // unfortunately, that's it.
                }
            } while (!(
                mst_labels[e_last] >= 0 &&  // currently always true
                !(skip_leaves && (is_leaf(mst_i[2*e_last+0]) || is_leaf(mst_i[2*e_last+1]))) &&
                std::min(
                    mst_cutsizes[e_last],
                    cluster_sizes[mst_labels[e_last]]-mst_cutsizes[e_last]
                ) >= min_cluster_size
            ));

            cut_edges[n_clusters_-1] = e_last;
            mst_labels[e_last]   = -1;
            mst_cutsizes[e_last] = -1;
            n_clusters_++;

            for (Py_ssize_t iv=0; iv <= 1; ++iv) {  // iv in {0,1} - go "left" and "right" along e_last
                Py_ssize_t v = mst_i[2*e_last+iv];

                if (iv == 1) labels[v] = n_clusters_-1;
                // else labels[v] stays the same

                Py_ssize_t tot = 1-(skip_leaves && is_leaf(v));
                for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
                    tot += visit(v, *pe);  // update vertex and edge labels and cutsizes
                }
                cluster_sizes[labels[v]] = tot;
            }
        }

        return n_clusters_;
    }


    /*! Propagate res with clustering results.
     *
     *  Unreachable vertices are assigned cluster -1.
     *
     *  @param c [out] array of length n
     */
    void get_labels(Py_ssize_t* c)
    {
        for (Py_ssize_t v=0; v<n; ++v)
            c[v] = labels[v];
    }


    /*! Get the indexes of the cut edges of the spanning tree that lead
     *  to n_clusters connected components.
     *
     *  @param res [out] array of length n_clusters-1
     */
    void get_cut_edges(Py_ssize_t* res)
    {
        for (size_t i=0; i<cut_edges.size(); ++i)
            res[i] = cut_edges[i];
    }
};


#endif
