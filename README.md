<a href="https://lumbermark.gagolewski.com/"><img src="https://www.gagolewski.com/_static/img/lumbermark.png" align="right" height="128" width="128" /></a>
# [**lumbermark**](https://lumbermark.gagolewski.com/) Package for R and Python

### *Lumbermark*: Resistant Clustering via Chopping Up Mutual Reachability Minimum Spanning Trees

![lumbermark for Python](https://github.com/gagolews/lumbermark/workflows/lumbermark%20for%20Python/badge.svg)
![lumbermark for R](https://github.com/gagolews/lumbermark/workflows/lumbermark%20for%20R/badge.svg)

**Keywords**: Lumbermark, clustering, Genie, HDBSCAN\*, DBSCAN, outliers,
minimum spanning tree, MST, density estimation, mutual reachability distance.


Refer to the package **homepage** at <https://lumbermark.gagolewski.com/>
for the reference manual, tutorials, examples, and benchmarks.

**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com/)


## About

*Lumbermark* is a fast and resistant divisive clustering algorithm which
identifies a specified number of clusters.

It iteratively chops off sizeable limbs that are joined by protruding segments
of a dataset's mutual reachability minimum spanning tree.

The use of a mutual reachability distance pulls peripheral points farther
away from each other.

When combined with the [**deadwood**](https://deadwood.gagolewski.com/) package,
it can act as an outlier detector.


## How to Install

### Python Version

To install from [PyPI](https://pypi.org/project/lumbermark), call:

```bash
pip3 install lumbermark  # python3 -m pip install lumbermark
```

*To learn more about Python, check out my open-access textbook*
[Minimalist Data Wrangling in Python](https://datawranglingpy.gagolewski.com/).


### R Version

To install from [CRAN](https://CRAN.R-project.org/package=lumbermark), call:

```r
install.packages("lumbermark")
```

*To learn more about R, check out my open-access textbook*
[Deep R Programming](https://deepr.gagolewski.com/).


### Other

The core functionality is implemented in the form of a C++ library.
It can thus be easily adapted for use in other projects.

New contributions are welcome, e.g., Julia, Matlab/GNU Octave wrappers.


## License

Copyright (C) 2025–2026 Marek Gagolewski <https://www.gagolewski.com/>

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License Version 3, 19
November 2007, published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
General Public License Version 3 for more details. You should have
received a copy of the License along with this program. If not, see
(https://www.gnu.org/licenses/).


## References

TODO

See **lumbermark**'s [homepage](https://lumbermark.gagolewski.com/) for more
references.
