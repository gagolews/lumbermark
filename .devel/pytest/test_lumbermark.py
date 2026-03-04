import numpy as np
import scipy.spatial.distance
import time
import gc
import lumbermark
import sklearn.datasets
import pandas as pd
import deadwood


def test_basic():
    n1, n2 = 100, 200
    np.random.seed(123)
    X = np.vstack((np.random.rand(n1, 2)+np.r_[-5, -5], np.random.rand(n2, 2)+np.r_[5, 5]))
    y_true = np.repeat([0, 1], [n1, n2])
    y_pred = lumbermark.Lumbermark(n_clusters=2).fit_predict(X)
    assert np.all(y_true==y_pred) or np.all(y_true==1-y_pred)

    y_pred = lumbermark.Lumbermark(n_clusters=2, M=10, min_cluster_size=10, min_cluster_factor=0.1, skip_leaves=False).fit_predict(X)
    assert np.all(y_true==y_pred) or np.all(y_true==1-y_pred)

    y_pred = lumbermark.Lumbermark(n_clusters=2, M=10, min_cluster_size=10, min_cluster_factor=0.1, skip_leaves=True).fit_predict(X)
    assert np.all(y_true==y_pred) or np.all(y_true==1-y_pred)


def test_iris():
    # duplicates!
    X, y_true = sklearn.datasets.load_iris(return_X_y=True)
    X = np.vstack((X,X))
    y_true = np.r_[y_true, y_true]
    for M in [0, 1, 2, 5, 10]:
        for skip_leaves in [True, False, "auto"]:
            L = lumbermark.Lumbermark(n_clusters=3, M=M, skip_leaves=skip_leaves).fit(X)
            assert L.n_clusters_ == 3
            assert np.all(L.labels_ >= 0)
            assert np.all(L.labels_ < 3)


def test_deadwood():
    np.random.seed(1234)
    n1, n2 = 1000, 250
    X = np.vstack((
        np.random.rand(n1, 2),
        np.random.rand(n2, 2)+[1.2, 0]
    ))

    for M in [0, 1, 2, 10, 25]:
        for skip_leaves in [True, False, "auto"]:
            G = lumbermark.Lumbermark(2, M=M, skip_leaves=skip_leaves).fit(X)
            y = G.labels_
            #genieclust.plots.plot_scatter(X, labels=y, asp=1, asp=1)
            #plt.show()

            D = deadwood.Deadwood()
            o = D.fit_predict(G)
            print(D.contamination_)
            #w = o.copy(); w[w>0] = y[w>0]
            # genieclust.plots.plot_scatter(X, labels=o, asp=1)
            # plt.show()
            assert (o[:n1]<0).mean() > 0.1
            assert (o[n1:]<0).mean() > 0.1


if __name__ == "__main__":
    test_basic()
    test_iris()
    test_deadwood()
