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



#' @title Lumbermark: Resistant Clustering via Splitting Mutual Reachability Minimum Spanning Trees
#'
#' @description
#' See \code{\link{lumbermark}()} for more details.
#'
#' @useDynLib lumbermark, .registration=TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom deadwood mst
#' @keywords internal
"_PACKAGE"
