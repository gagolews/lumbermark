import numpy as np
import scipy.spatial.distance
import time
import gc
import lumbermark


def test_lumbermark():
    n1, n2 = 100, 200
    np.random.seed(123)
    X = np.vstack((np.random.rand(n1, 2)+np.r_[-5, -5], np.random.rand(n2, 2)+np.r_[5, 5]))
    y_true = np.repeat([0, 1], [n1, n2])
    y_pred = lumbermark.Lumbermark(n_clusters=2).fit_predict(X)
    assert np.all(y_true == y_pred)


if __name__ == "__main__":
    test_lumbermark()
