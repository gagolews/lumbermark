



# R Examples

*To learn more about R, check out my open-access textbook*
[Deep R Programming](https://deepr.gagolewski.com/).


## How to Install

To install the package from [CRAN](https://CRAN.R-project.org/package=lumbermark), call:


```r
install.packages("lumbermark")
```



## Basic Use

::::{note}
This section is a work in progress.  In the meantime, take a look at
the documentation of the [lumbermark](../rapi/lumbermark) function
and the paper {cite}`Gagolewski2026:lumbermark`.
::::

[wut/z2](https://clustering-benchmarks.gagolewski.com/) is an example
dataset with five clusters of rather non-homogeneous densities.
*Lumbermark* separates them correctly with no need for further parameter
tuning:


``` r
Z2 <- as.matrix(read.table("z2.data.gz"))
library("lumbermark")
lumbermark_labels <- lumbermark(Z2, 5)
plot(Z2, asp=1, ann=FALSE, col=lumbermark_labels)
```

(fig:r_z2_dataset)=
```{figure} r-figures/r_z2_dataset-1.*
The z2 dataset
```



## Outlier Detection in the Case of Clusters of Heterogeneous Densities

The recently-developed [*Deadwood*](https://deadwood.gagolewski.com/)
outlier detection algorithm works quite well in the case of clusters
of similar densities. However, we can combine it with *Lumbermark*
to detect outliers in each detected cluster separately.



``` r
par(mfrow=c(1, 2))

library("deadwood")

is_outlier_homo   <- deadwood(Z2)
plot(Z2, asp=1, xlab=NA, ylab=NA,
    col=c("#00000055","#ff333333")[is_outlier_homo+1], main="Deadwood")

is_outlier_hetero <- deadwood(lumbermark_labels)
plot(Z2, asp=1, xlab=NA, ylab=NA,
    col=c("#00000055","#ff333333")[is_outlier_hetero+1], main="Deadwood+Lumbermark")
```

(fig:r_z2_deadwood)=
```{figure} r-figures/r_z2_deadwood-1.*
Outlier detection of the *z2* dataset
```


