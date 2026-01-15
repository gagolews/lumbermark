import numpy as np
import scipy.spatial.distance
import time
import gc
import lumbermark
import sklearn.datasets
import pandas as pd
import genieclust


def test_basic():
    n1, n2 = 100, 200
    np.random.seed(123)
    X = np.vstack((np.random.rand(n1, 2)+np.r_[-5, -5], np.random.rand(n2, 2)+np.r_[5, 5]))
    y_true = np.repeat([0, 1], [n1, n2])
    y_pred = lumbermark.Lumbermark(n_clusters=2).fit_predict(X)
    assert genieclust.compare_partitions.normalized_pivoted_accuracy(y_true, y_pred)>1-1e-9

    y_pred = lumbermark.Lumbermark(n_clusters=2, M=10, min_cluster_size=10, min_cluster_factor=0.1).fit_predict(X)
    assert genieclust.compare_partitions.normalized_pivoted_accuracy(y_true, y_pred)>1-1e-9


def test_iris():
    # duplicates!
    X, y_true = sklearn.datasets.load_iris(return_X_y=True)
    X = np.vstack((X,X))
    y_true = np.r_[y_true, y_true]
    L = lumbermark.Lumbermark(n_clusters=3).fit(X)
    assert L.n_clusters_ == 3
    assert np.all(L.labels_ >= 0)
    assert np.all(L.labels_ < 3)

if __name__ == "__main__":
    test_basic()
