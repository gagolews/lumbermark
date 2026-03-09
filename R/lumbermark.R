# This file is part of the lumbermark package for R.

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


#' @title Lumbermark: Fast and Robust Clustering
#'
#' @description
#' Lumbermark is a fast and robust divisive clustering algorithm
#' which identifies a specified number of clusters.
#'
#' It iteratively chops off sizeable limbs that are joined by protruding
#' segments of a dataset's mutual reachability minimum spanning tree.
#'
#' The use of a mutual reachability distance (\eqn{M>1}; Campello et al., 2013)
#' pulls peripheral points farther away from each other.
#' This way, Lumbermark gives an alternative to the HDBSCAN* algorithm
#' that is able to detect a predefined number of clusters and indicate
#' outliers (via \pkg{deadwood}; see Gagolewski, 2026).
#'
#'
#' @details
#' As with all distance-based methods (this includes k-means and DBSCAN as well),
#' applying data preprocessing and feature engineering techniques
#' (e.g., feature scaling, feature selection, dimensionality reduction)
#' might lead to more meaningful results.
#'
#' If \code{d} is a numeric matrix or an object of class \code{dist},
#' \code{\link[deadwood]{mst}()} will be called to compute an MST, which generally
#' takes at most \eqn{O(n^2)} time. However, by default, a faster algorithm
#' based on K-d trees is selected automatically for low-dimensional Euclidean
#' spaces; see \code{\link[quitefastmst]{mst_euclid}} from
#' the \pkg{quitefastmst} package.
#'
#' Once a minimum spanning tree is determined, the Lumbermark algorithm runs in
#' \eqn{O(kn)} time.  If you want to test different parameters or \eqn{k}s,
#' it is best to compute the MST explicitly beforehand.
#'
#'
#' @param d a numeric matrix (or an object coercible to one,
#'     e.g., a data frame with numeric-like columns) or an
#'     object of class \code{dist} (see \code{\link[stats]{dist}}),
#'     or an object of class \code{mst} (see \code{\link[deadwood]{mst}})
#'
#' @param min_cluster_size integer;
#'     minimal cluster size
#'
#' @param min_cluster_factor numeric value in (0,1); output cluster sizes will
#'     not be smaller than \code{min_cluster_factor*n/k}
#'
#' @param skip_leaves logical; whether the MST leaves should be omitted
#'     from cluster size counting
#'
#' @param distance metric used to compute the linkage, one of:
#'     \code{"euclidean"} (synonym: \code{"l2"}),
#'     \code{"manhattan"} (a.k.a. \code{"l1"} and \code{"cityblock"}),
#'     \code{"cosine"}
#'
#' @param verbose logical; whether to print diagnostic messages
#'     and progress information
#'
#' @param k integer; the desired number of clusters to detect
#'
#' @param M integer; smoothing factor; \eqn{M \leq 1} gives the selected
#'    \code{distance}; otherwise, the mutual reachability distance is used
#'
#' @param ... further arguments passed to \code{\link[deadwood]{mst}()}
#'
#'
#' @return
#' \code{lumbermark()} returns an object of class \code{mstclust}, which defines
#' a \eqn{k}-partition, i.e., a vector whose \eqn{i}-th element denotes
#' the \eqn{i}-th input point's cluster label between 1 and \eqn{k}.
#'
#' The \code{mst} attribute gives the computed minimum
#' spanning tree which can be reused in further calls to the functions
#' from \pkg{genieclust}, \pkg{lumbermark}, and \pkg{deadwood}.
#'
#' The \code{cut_edges} attribute gives the \eqn{k-1}
#' indexes of the MST edges whose omission leads to the requested
#' \eqn{k}-partition (connected components of the resulting spanning forest).
#'
#'
#' @seealso
#' \code{\link[deadwood]{mst}()} for the minimum spanning tree routines
#'
#'
#' @references
#' M. Gagolewski, lumbermark, in preparation, 2026
#'
#' R.J.G.B. Campello, D. Moulavi, J. Sander,
#' Density-based clustering based on hierarchical density estimates,
#' \emph{Lecture Notes in Computer Science} 7819, 2013, 160-172,
#' \doi{10.1007/978-3-642-37456-2_14}
#'
#' M. Gagolewski, A. Cena, M. Bartoszuk, Ł. Brzozowski,
#' Clustering with minimum spanning trees: How good can it be?,
#' \emph{Journal of Classification} 42, 2025, 90-112,
#' \doi{10.1007/s00357-024-09483-1}
#'
#' M. Gagolewski, deadwood, in preparation, 2026
#'
#' M. Gagolewski, quitefastmst, in preparation, 2026
#'
#'
#' @examples
#' library("datasets")
#' data("iris")
#' X <- jitter(as.matrix(iris[3:4]))
#' y_pred <- lumbermark(X, k=3)
#' y_test <- as.integer(iris[,5])
#' plot(X, col=y_pred, pch=y_test, asp=1, las=1)
#'
#' # detect 3 clusters and find outliers with Deadwood
#' library("deadwood")
#' y_pred2 <- lumbermark(X, k=3)
#' plot(X, col=y_pred2, asp=1, las=1)
#' is_outlier <- deadwood(y_pred2)
#' points(X[!is_outlier, ], col=y_pred2[!is_outlier], pch=16)
#'
#' @rdname lumbermark
#' @export
lumbermark <- function(d, ...)
{
    UseMethod("lumbermark")
}


#' @export
#' @rdname lumbermark
#' @method lumbermark default
lumbermark.default <- function(
    d,
    k,
    min_cluster_size=10,
    min_cluster_factor=0.25,
    skip_leaves=(M>0L),
    M=0L,
    distance=c("euclidean", "l2", "manhattan", "cityblock", "l1", "cosine"),
    verbose=FALSE,
    ...
) {
    distance <- match.arg(distance)
    lumbermark.mst(
        mst(d, M=M, distance=distance, verbose=verbose, ...),
        k=k,
        min_cluster_size=min_cluster_size,
        min_cluster_factor=min_cluster_factor,
        skip_leaves=skip_leaves,
        verbose=verbose
    )
}


#' @export
#' @rdname lumbermark
#' @method lumbermark dist
lumbermark.dist <- function(
    d,
    k,
    min_cluster_size=10,
    min_cluster_factor=0.25,
    skip_leaves=(M>0L),
    M=0L,
    verbose=FALSE,
    ...
) {
    lumbermark.mst(
        mst(d, M=M, verbose=verbose, ...),
        k=k,
        min_cluster_size=min_cluster_size,
        min_cluster_factor=min_cluster_factor,
        skip_leaves=skip_leaves,
        verbose=verbose
    )
}


#' @export
#' @rdname lumbermark
#' @method lumbermark mst
lumbermark.mst <- function(
    d,
    k,
    min_cluster_size=10,
    min_cluster_factor=0.25,
    skip_leaves=TRUE,
    verbose=FALSE,
    ...
) {
    min_cluster_factor <- as.double(min_cluster_factor)[1]
    stopifnot(min_cluster_factor > 0.0, min_cluster_factor < 1.0)

    min_cluster_size <- as.integer(min_cluster_size)[1]
    stopifnot(min_cluster_size > 0)

    skip_leaves <- !identical(skip_leaves, FALSE)

    verbose <- !identical(verbose, FALSE)

    clusters <- .lumbermark(
        d,
        k=k,
        min_cluster_size=min_cluster_size,
        min_cluster_factor=min_cluster_factor,
        skip_leaves=skip_leaves,
        verbose=verbose
    )

    stopifnot(length(attr(clusters, "cut_edges")) == k-1)

    structure(
        clusters,
        names=attr(d, "Labels"),
        mst=d,
        #cut_edges=cut_edges,  already there
        class="mstclust"
    )
}


registerS3method("lumbermark", "default", "lumbermark.default")
registerS3method("lumbermark", "dist",    "lumbermark.dist")
registerS3method("lumbermark", "mst",     "lumbermark.mst")
