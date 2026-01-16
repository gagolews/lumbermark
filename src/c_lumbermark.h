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

// special vertex/edge markers - must be negative!
#define LUMBERMARK_UNSET     (std::numeric_limits<Py_ssize_t>::min())
#define LUMBERMARK_CUTEDGE   (LUMBERMARK_UNSET+1)
#define LUMBERMARK_NOISEEDGE (LUMBERMARK_UNSET+2)



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

    const FLOAT* mst_d;  //<! m edge weights, sorted increasingly
    const Py_ssize_t* mst_i;  //<! 2*m edge definitions
    Py_ssize_t m;        //<! number of edges, must be n-1
    Py_ssize_t n;        //<! number of points

    const Py_ssize_t* cumdeg;  // nullable or length n+1; see Cgraph_vertex_incidences in 'deadwood'
    const Py_ssize_t* inc;  // nullable or length 2*m; see Cgraph_vertex_incidences in 'deadwood'


    // auxiliary data for generating clustering results:
    std::vector<Py_ssize_t> mst_labels;  //<! edge labels, size m
    std::vector<Py_ssize_t> labels;  //<! node labels, size n, in 1..n_clusters and -1..-n_clusters (outliers/noise points)
    std::vector<Py_ssize_t> mst_cutsizes;  //<! size m, each pair gives the sizes of the clusters that are formed when we cut out the corresponding edge

    std::vector<Py_ssize_t> cluster_sizes; //<!  size n_clusters+1
    std::vector<Py_ssize_t> cut_edges; //<!  size n_clusters-1


    /*! vertex visitor (1st pass):
     *  going from v, visits w and then all its neighbours, mst_i[e,:] = {v,w};
     *  checks if the graph is acyclic
     */
    Py_ssize_t visit1(Py_ssize_t v, Py_ssize_t e)
    {
        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];
        //     so v == mst_i[2*e+iv]

        LUMBERMARK_ASSERT(e >= 0 && e < m);
        LUMBERMARK_ASSERT(v >= 0 && v < n);
        LUMBERMARK_ASSERT(w >= 0 && w < n);
        LUMBERMARK_ASSERT(labels[v] == 1);
        LUMBERMARK_ASSERT(mst_labels[e] == LUMBERMARK_UNSET);
        LUMBERMARK_ASSERT(labels[w] == LUMBERMARK_UNSET);

        Py_ssize_t tot = 0;
        labels[w] = 1;
        mst_labels[e] = 1;
        tot++;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) tot += visit1(w, *pe);
        }

        mst_cutsizes[e] = tot;

        return tot;
    }


    /*! vertex visitor (k-th pass):
     *  going from v, visits w and then all its neighbours, mst_i[e,:] = {v,w};
     *  marks them as members of the c-th cluster. */
    Py_ssize_t visitk(Py_ssize_t v, Py_ssize_t e, Py_ssize_t c)
    {
        if (mst_labels[e] == LUMBERMARK_CUTEDGE)
            return 0;

        Py_ssize_t iv = (Py_ssize_t)(mst_i[2*e+1]==v);
        Py_ssize_t w = mst_i[2*e+(1-iv)];
        //     so v == mst_i[2*e+iv]

        Py_ssize_t tot = 0;
        LUMBERMARK_ASSERT(c > 0);
        labels[w] = c;
        mst_labels[e] = c;
        tot++;

        for (const Py_ssize_t* pe = inc+cumdeg[w]; pe != inc+cumdeg[w+1]; pe++) {
            if (*pe != e) tot += visitk(w, *pe, c);
        }

        mst_cutsizes[e] = tot;

        return tot;
    }


    void init_labels()
    {
        labels.resize(n);
        for (Py_ssize_t v=0; v<n; ++v)
            labels[v] = LUMBERMARK_UNSET;

        mst_labels.resize(m);
        for (Py_ssize_t e=0; e<m; ++e)
            mst_labels[e] = LUMBERMARK_UNSET;

        mst_cutsizes.resize(m);

        // ensure that the graph is acyclic:
        // visit all nodes starting from a vertex incident to edge 0
        Py_ssize_t v = mst_i[2*0+0];
        Py_ssize_t tot = 1;
        labels[v] = 1;
        for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
            tot += visit1(v, *pe);
        }
        cluster_sizes[1] = tot;

        // ensure all m edges are reachable:
        for (Py_ssize_t e=0; e<m; ++e)
            LUMBERMARK_ASSERT(mst_labels[e] != LUMBERMARK_UNSET);
    }


public:
    CLumbermark() : CLumbermark(nullptr, nullptr, 0, false, nullptr, nullptr) { }

    CLumbermark(
        const FLOAT* mst_d, const Py_ssize_t* mst_i, Py_ssize_t m, Py_ssize_t n,
        const Py_ssize_t* cumdeg, const Py_ssize_t* inc
    ) :
        mst_d(mst_d), mst_i(mst_i), m(m), n(n), cumdeg(cumdeg), inc(inc)
    {
        if (n == 0) return;

        if (m != n-1)
            throw std::domain_error("m != n-1");

        for (Py_ssize_t e=1; e<m; ++e) {
            // check if edge weights are sorted increasingly
            LUMBERMARK_ASSERT(mst_d[e-1] <= mst_d[e]);
        }
    }


    /*! Run the Lumbermark algorithm
     *
     * @param n_clusters number of clusters to find
     * @param min_cluster_size minimal cluster size
     * @param min_cluster_factor output cluster sizes won't be smaller than
     *    min_cluster_factor/n_clusters*n_points (noise points excluding)
     *
     * @return number of clusters detected (can be smaller than the requested one)
     */
    Py_ssize_t compute(
        Py_ssize_t n_clusters, Py_ssize_t min_cluster_size, FLOAT min_cluster_factor
    ) {
        LUMBERMARK_ASSERT(n > 2);

        if (n_clusters < 1)
            throw std::domain_error("n_clusters must be >= 1");

        Py_ssize_t n_skip = 0;  // count unreachable vertices
        for (Py_ssize_t v=0; v<n; ++v)
            if (cumdeg[v+1]-cumdeg[v] <= 0) n_skip++;

        if (n_skip > 0)
            throw std::domain_error("there are unreachable vertices");

        min_cluster_size = std::max(
            min_cluster_size,
            (Py_ssize_t)(min_cluster_factor*(n-n_skip)/n_clusters)
        );

        cut_edges.resize(n_clusters-1);
        cluster_sizes.resize(n_clusters+1);  // 1-based

        init_labels();

        Py_ssize_t n_clusters_ = 1;
        Py_ssize_t e_last = m;  // edges are consumed in decreasing order

        while (n_clusters_ < n_clusters)
        {
            do {
                e_last--;
                if (e_last < 0) {
                    cut_edges.resize(n_clusters_-1);
                    return n_clusters_;  // unfortunately, that's it.
                }
            } while (!(
                mst_labels[e_last] > 0 &&
                std::min(
                    mst_cutsizes[e_last],
                    cluster_sizes[mst_labels[e_last]]-mst_cutsizes[e_last]
                ) >= min_cluster_size
            ));

            cut_edges[n_clusters_-1] = e_last;
            mst_labels[e_last]   = LUMBERMARK_CUTEDGE;
            mst_cutsizes[e_last] = LUMBERMARK_UNSET;
            n_clusters_++;
            // GENIECLUST_PRINT("***%d***\n", n_clusters_);

            for (int iv=0; iv <= 1; ++iv) {  // iv in {0,1}
                Py_ssize_t v = mst_i[2*e_last+iv];

                // if (skip_leaves) {
                //     LUMBERMARK_ASSERT(deg[v] > 1);
                //     if (deg[v] == 2) {
                //         // mark v as incident to a cut edge and a noise edge,
                //         // because it's a leaf in the newly-formed cluster
                //         Py_ssize_t e = inc[v][0];
                //         if (e == e_last) e = inc[v][1];
                //         mst_labels[e] = LUMBERMARK_NOISEEDGE;
                //         if (mst_i[2*e+0] == v) v = mst_i[2*e+1];
                //         else v = mst_i[2*e+0];
                //     }
                // }


                if (iv == 1) labels[v] = n_clusters_;
                // else labels[v] stays the same
                LUMBERMARK_ASSERT(labels[v] > 0);

                Py_ssize_t tot = 1;
                for (const Py_ssize_t* pe = inc+cumdeg[v]; pe != inc+cumdeg[v+1]; pe++) {
                    tot += visitk(v, *pe, labels[v]);
                }

                cluster_sizes[labels[v]] = tot;
            }
        }

        return n_clusters_;
    }


    // /*! Set res[i] to true if vertex i is unreachable.
    //  *
    //  *  @param res [out] array of length n
    //  */
    // void get_is_unreachable(bool* res) const
    // {
    //     for (Py_ssize_t v=0; v<n; ++v) {
    //         //LUMBERMARK_ASSERT(labels[v] != LUMBERMARK_UNSET);
    //         LUMBERMARK_ASSERT(labels[v] != 0);
    //         res[v] = (bool)(labels[v] < 0);
    //     }
    // }



    /*! Propagate res with clustering results.
     *
     *  Unreachable vertices are assigned cluster -1.
     *
     *  @param res [out] array of length n
     */
    void get_labels(Py_ssize_t* res)
    {
        for (Py_ssize_t v=0; v<n; ++v) {
            //LUMBERMARK_ASSERT(labels[v] != LUMBERMARK_UNSET);
            LUMBERMARK_ASSERT(labels[v] != 0);
            if (labels[v] > 0) res[v] = labels[v]-1;
            else res[v] = -1; //(-labels[v])-1;
        }
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
