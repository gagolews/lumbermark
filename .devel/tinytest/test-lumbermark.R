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
expect_true(all(tabulate(y) >= 0.15*n/2))

y <- lumbermark(X, k=5)
expect_true(length(y) == n)
expect_true(length(attr(y, "cut_edges")) == 4)
expect_true(min(y) >= 1, max(y) <= 5)
expect_true(all(tabulate(y) > 0.15*n/5))

y <- lumbermark(X, k=5, nested=TRUE)
expect_true(length(y) == n)
expect_true(length(attr(y, "cut_edges")) == 4)
expect_true(min(y) >= 1, max(y) <= 5)
expect_true(all(tabulate(y) > 0.15*n/5))
