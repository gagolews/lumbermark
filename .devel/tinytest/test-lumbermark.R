library("tinytest")
library("lumbermark")

#set.seed(123)
n <- 1000
d <- rpois(1, 10)+2
X <- matrix(rnorm(n*d), nrow=n)

y <- lumbermark(X, k=2)
expect_true(length(y) == n)
expect_true(length(attr(y, "cut_edges")) == 1)
expect_true(min(y) >= 1, max(y) <= 2)
expect_true(all(tabulate(y) > n/10))
