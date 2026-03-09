# *Lumbermark*: Resistant Clustering via Chopping Up Mutual Reachability Minimum Spanning Trees

::::{image} _static/img/lumbermark_toy_example.png
:class: img-right-align-always
:alt: Lumbermark
:width: 128px
::::


**Keywords**: Lumbermark, clustering, HDBSCAN\*, DBSCAN, outliers,
minimum spanning tree, MST, density estimation, mutual reachability distance.


*Lumbermark* is a fast and resistant divisive clustering algorithm which
identifies a specified number of clusters.

It iteratively chops off sizeable limbs that are joined by protruding segments
of a dataset's mutual reachability minimum spanning tree.

The use of a mutual reachability distance pulls peripheral points farther
away from each other.

When combined with the [**deadwood**](https://deadwood.gagolewski.com/) package,
it can act as an outlier detector.


## Contributing

**lumbermark** is distributed under the open source GNU AGPL v3 license.
Its source code can be downloaded from [GitHub](https://github.com/gagolews/lumbermark).

The Python version is available from [PyPI](https://pypi.org/project/lumbermark).
The R version can be fetched from [CRAN](https://CRAN.R-project.org/package=lumbermark).

The core functionality is implemented in the form of a C++ library.
It can thus be easily adapted for use in other environments.
New contributions are welcome, e.g., Julia, Matlab/GNU Octave wrappers.


**Author and Maintainer**: [Marek Gagolewski](https://www.gagolewski.com/)



::::{toctree}
:maxdepth: 2
:caption: lumbermark
:hidden:

About <self>
Author <https://www.gagolewski.com/>
Source Code (GitHub) <https://github.com/gagolews/lumbermark>
Bug Tracker and Feature Suggestions <https://github.com/gagolews/lumbermark/issues>
PyPI Entry <https://pypi.org/project/lumbermark>
CRAN Entry <https://CRAN.R-project.org/package=lumbermark>
::::



::::{toctree}
:maxdepth: 1
:caption: Python API
:hidden:

weave/python
weave/sklearn_toy_example
pythonapi
::::


::::{toctree}
:maxdepth: 1
:caption: R API
:hidden:

weave/r
rapi
::::


::::{toctree}
:maxdepth: 1
:caption: Other
:hidden:


quitefastmst <https://quitefastmst.gagolewski.com/>
deadwood <https://deadwood.gagolewski.com/>
genieclust <https://genieclust.gagolewski.com/>
Clustering Benchmarks <https://clustering-benchmarks.gagolewski.com/>
Minimalist Data Wrangling in Python <https://datawranglingpy.gagolewski.com/>
Deep R Programming <https://deepr.gagolewski.com/>
news
z_bibliography
::::


<!--
Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
-->
