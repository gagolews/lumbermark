/*  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
 *
 *  Copyleft (C) 2025-2025, Marek Gagolewski <https://www.gagolewski.com>
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



/*! Compute the degree of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *  NOTE: We have the same function in genieclust+lumbermark
 *
 * @param ind c_contiguous matrix of size m*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n.
 *     Edges with ind[i,0] < 0 or ind[i,1] < 0 are purposely ignored
 * @param m number of edges (rows in ind)
 * @param n number of vertices
 * @param deg [out] array of size n, where
 *     deg[i] will give the degree of the i-th vertex.
 */
void Cget_graph_node_degrees(
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

        if (u < 0) {
            LUMBERMARK_ASSERT(v < 0);
            continue; // represents a no-edge -> ignore
        }
        LUMBERMARK_ASSERT(v >= 0);

        if (u >= n || v >= n)
            throw std::domain_error("All elements must be < n");
        if (u == v)
            throw std::domain_error("Self-loops are not allowed");

        deg[u]++;
        deg[v]++;
    }
}



/*! Compute the incidence list of each vertex in an undirected graph
 *  over a vertex set {0,...,n-1}.
 *
 *  NOTE: We have the same function in genieclust+lumbermark
 *
 * @param ind c_contiguous matrix of size m*2,
 *     where {ind[i,0], ind[i,1]} is the i-th edge with ind[i,j] < n.
 *     Edges with ind[i,0] < 0 or ind[i,1] < 0 are purposely ignored
 * @param m number of edges (rows in ind)
 * @param n number of vertices
 * @param deg array of size n, where deg[i] gives the degree of the i-th vertex
 * @param data [out] a data buffer of length 2*m, provides data for adj
 * @param adj [out] an array of length n+1, where adj[i] will be an array
 *     of length deg[i] giving the edges incident on the i-th vertex;
 *     adj[n] is a sentinel element
 */
void Cget_graph_node_inclists(
    const Py_ssize_t* ind,
    const Py_ssize_t m,
    const Py_ssize_t n,
    const Py_ssize_t* deg,
    Py_ssize_t* data,
    Py_ssize_t** inc
) {
    Py_ssize_t cumdeg = 0;
    inc[0] = data;
    for (Py_ssize_t i=0; i<n; ++i) {
        inc[i+1] = data+cumdeg;
        cumdeg += deg[i];
    }

    LUMBERMARK_ASSERT(cumdeg <= 2*m);

    for (Py_ssize_t i=0; i<m; ++i) {
        Py_ssize_t u = ind[2*i+0];
        Py_ssize_t v = ind[2*i+1];
        if (u < 0 || v < 0)
            continue; // represents a no-edge -> ignore

#ifdef DEBUG
        if (u >= n || v >= n)
            throw std::domain_error("All elements must be < n");
        if (u == v)
            throw std::domain_error("Self-loops are not allowed");
#endif

        *(inc[u+1]) = i;
        ++(inc[u+1]);

        *(inc[v+1]) = i;
        ++(inc[v+1]);
    }

#ifdef DEBUG
    cumdeg = 0;
    inc[0] = data;
    for (Py_ssize_t i=0; i<n; ++i) {
        LUMBERMARK_ASSERT(inc[i] == data+cumdeg);
        cumdeg += deg[i];
    }
#endif
}




/*!  Lumbermark: A Robust Divisive Clustering Algorithm based on Spanning Trees
 *
 *   Splits a tree into a given number of connected components.
 *   Unreachable vertices are treated as outliers/noise points.
 *
 *
 *   References
 *   ==========
 *
 *   Gagolewski, M., TODO, 2025
 */
template <class FLOAT>
class CLumbermark {
protected:

    const FLOAT* mst_d;  //<! m edge weights, sorted increasingly

    /*! m edges of the spanning tree given by c_contiguous m*2 indices;
     ** (-1, -1) denotes a no-edge and will be ignored;
     normally, m=n-1, but the tree does not have to span all the vertices;
     the unreachable nodes are treated as outliers
     */
    const Py_ssize_t* mst_i;


    Py_ssize_t m;        //<! number of edges
    Py_ssize_t n;        //<! number of points

    std::vector<Py_ssize_t> deg;  //<! deg[i] denotes the degree of the i-th vertex

    std::vector<Py_ssize_t*> inc;  //<! inc[i] is a length-deg[i] array of edges incident on the i-th vertex; inc's length is n+1 (inc[n] is a sentinel element)
    std::vector<Py_ssize_t> _incdata;  //<! the underlying data buffer for inc


    // auxiliary data for generating clustering results:
    std::vector<Py_ssize_t> mst_labels;  //<! edge labels, size m
    std::vector<Py_ssize_t> labels;  //<! node labels, size n, in 1..n_clusters and -1..-n_clusters (outliers/noise points)
    std::vector<Py_ssize_t> mst_cutsizes;  //<!  size m*2, each pair gives the sizes of the clusters that are formed when we cut out the corresponding edge

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

        for (Py_ssize_t* e2 = inc[w]; e2 != inc[w+1]; e2++) {
            if (*e2 != e) tot += visit1(w, *e2);
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

        for (Py_ssize_t* e2 = inc[w]; e2 != inc[w+1]; e2++) {
            if (*e2 != e) tot += visitk(w, *e2, c);
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
        for (Py_ssize_t* e2 = inc[v]; e2 != inc[v+1]; e2++) {
            tot += visit1(v, *e2);
        }
        cluster_sizes[1] = tot;

        // ensure all m edges are reachable:
        for (Py_ssize_t e=0; e<m; ++e)
            LUMBERMARK_ASSERT(mst_labels[e] != LUMBERMARK_UNSET);
    }


public:
    CLumbermark() : CLumbermark(NULL, NULL, 0, false) { }

    CLumbermark(const FLOAT* mst_d, const Py_ssize_t* mst_i, Py_ssize_t m, Py_ssize_t n) :
        mst_d(mst_d), mst_i(mst_i), m(m), n(n)
    {
        if (n == 0) return;

        for (Py_ssize_t e=0; e<m; ++e) {
            //LUMBERMARK_ASSERT(mst_i[2*e+0] >= 0);  // TODO: add to cutlist
            //LUMBERMARK_ASSERT(mst_i[2*e+1] >= 0);  // TODO: add to cutlist

            // check if edge weights are sorted increasingly
            LUMBERMARK_ASSERT(e == 0 || mst_d[e-1] <= mst_d[e]);
        }

        deg.resize(n);
        _incdata.resize(2*m);
        inc.resize(n+1);  // +sentinel

        // set up this->deg:
        Cget_graph_node_degrees(mst_i, m, n, /*out:*/this->deg.data());

        //for (Py_ssize_t v=0; v<n; ++v) {
        //    LUMBERMARK_ASSERT(deg[v] > 0);  // doesn't hold if the graph is not connected
        //}

        Cget_graph_node_inclists(
            mst_i, m, n, this->deg.data(),
            /*out:*/this->_incdata.data(), /*out:*/this->inc.data()
        );
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
            if (deg[v] <= 0) n_skip++;

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
                    return n_clusters_; // unfortunately, that's it.
                }

                // if (mst_labels[e_last] > 0)
                //     GENIECLUST_PRINT("%3d: label=%3d cutsize=%3d clustsize=%3d\n",
                //        e_last,
                //        mst_labels[e_last],
                //        mst_cutsizes[e_last],
                //        cluster_sizes[mst_labels[e_last]]);
                // else
                //     GENIECLUST_PRINT("%3d\n", e_last);

                // NOTE: we could be taking into account the fact that a node
                // incident to a cut edge might become a leaf (size adjustment),
                // but it's too much of a hassle; the benefits are questionable
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
                for (Py_ssize_t* e2 = inc[v]; e2 != inc[v+1]; e2++) {
                    tot += visitk(v, *e2, labels[v]);
                }

                cluster_sizes[labels[v]] = tot;
            }
        }

        return n_clusters_;
    }


    /*! Set res[i] to true if vertex i is unreachable.
     *
     *  @param res [out] array of length n
     */
    void get_is_unreachable(bool* res) const
    {
        for (Py_ssize_t v=0; v<n; ++v) {
            //LUMBERMARK_ASSERT(labels[v] != LUMBERMARK_UNSET);
            LUMBERMARK_ASSERT(labels[v] != 0);
            res[v] = (bool)(labels[v] < 0);
        }
    }



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
