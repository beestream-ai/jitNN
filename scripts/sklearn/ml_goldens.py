"""Reference results from scikit-learn for jitNN's classical machine-learning estimators.

Each case stores its dataset and what scikit-learn computes from it; MlAccelTest (jitnn-nn) fits
the jitNN estimator on the same data and compares. Written to jitnn-nn/src/test/resources/ml/:

  manifest.txt   one line per array:  <case> <name> <f32|i32> <offset> <count> <dims...>
  data.bin       the arrays, little-endian, back to back

usage: python ml_goldens.py <output dir>
"""
import os
import sys

import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

rng = np.random.default_rng(42)
arrays = []  # (case, name, array)


def put(case, **kw):
    for name, a in kw.items():
        a = np.asarray(a)
        arrays.append((case, name, a.astype(np.int32) if a.dtype.kind in "iub" else a.astype(np.float32)))


def blobs(n, d, centres, std=1.0, seed=0):
    x, y = datasets.make_blobs(n_samples=n, n_features=d, centers=centres, cluster_std=std, random_state=seed)
    return x.astype(np.float32), y


# k-means from given centres, and k-means++ restarts
x, _ = blobs(300, 2, 3, seed=1)
init = x[:3].copy()
km = KMeans(3, init=init, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(x)
put("kmeans", x=x, init=init, centres=km.cluster_centers_, labels=km.labels_, inertia=[km.inertia_])
x, _ = blobs(500, 4, 5, std=1.5, seed=2)
km = KMeans(5, n_init=10, random_state=0).fit(x)
put("kmeans_pp", x=x, inertia=[km.inertia_])

# k nearest neighbours
x, y = blobs(260, 5, 3, std=3.0, seed=3)
xt, yt, xq = x[:200], y[:200], x[200:]
put("knn", x=xt, y=yt, q=xq, pred=KNeighborsClassifier(5).fit(xt, yt).predict(xq),
    proba=KNeighborsClassifier(5).fit(xt, yt).predict_proba(xq),
    reg=KNeighborsRegressor(5).fit(xt, xt[:, 0] * 2 - xt[:, 1]).predict(xq), target=xt[:, 0] * 2 - xt[:, 1])

# PCA
z = rng.normal(size=(120, 3))
x = (z @ rng.normal(size=(3, 6)) + 0.1 * rng.normal(size=(120, 6)) + 3).astype(np.float32)
pca = PCA(3).fit(x)
put("pca", x=x, components=pca.components_, variance=pca.explained_variance_, ratio=pca.explained_variance_ratio_, transformed=pca.transform(x))

# least squares and ridge
x = rng.normal(size=(80, 5)).astype(np.float32)
y = (x @ np.array([1.5, -2.0, 0.5, 0.0, 3.0]) + 0.7 + 0.1 * rng.normal(size=80)).astype(np.float32)
lr = LinearRegression().fit(x, y)
rd = Ridge(alpha=5.0).fit(x, y)
put("linear", x=x, y=y, coef=lr.coef_, intercept=[lr.intercept_], ridge_coef=rd.coef_, ridge_intercept=[rd.intercept_])

# logistic regression, binary and multinomial; linear SVM
for name, k, seed in (("logreg_binary", 2, 4), ("logreg_multi", 3, 5)):
    x, y = blobs(300, 4, k, std=2.5, seed=seed)
    m = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(x, y)
    put(name, x=x, y=y, coef=m.coef_, intercept=m.intercept_, proba=m.predict_proba(x))
x, y = blobs(300, 4, 3, std=2.5, seed=6)
svc = LinearSVC(C=1.0, tol=1e-8, max_iter=200000).fit(x, y)
put("svc", x=x, y=y, coef=svc.coef_, intercept=svc.intercept_, pred=svc.predict(x))

# naive Bayes
x, y = blobs(150, 4, 3, std=1.5, seed=7)
nb = GaussianNB().fit(x, y)
put("gaussian_nb", x=x, y=y, theta=nb.theta_, var=nb.var_, proba=nb.predict_proba(x))
y = rng.integers(0, 3, 100)
x = rng.poisson(lam=(1 + y[:, None] * np.arange(10)[None, :] % 4), size=(100, 10)).astype(np.float32)
nb = MultinomialNB(alpha=1.0).fit(x, y)
put("multinomial_nb", x=x, y=y, log_prob=nb.feature_log_prob_, proba=nb.predict_proba(x))

# trees: held-out accuracy and R^2 of scikit-learn's models
x, y = datasets.make_classification(1000, 8, n_informative=5, n_redundant=1, n_classes=3, random_state=8)
x = x.astype(np.float32)
xa, ya, xb, yb = x[:800], y[:800], x[800:], y[800:]
put("trees_classification", x=xa, y=ya, xt=xb, yt=yb,
    tree=[accuracy_score(yb, DecisionTreeClassifier(max_depth=6, random_state=0).fit(xa, ya).predict(xb))],
    forest=[accuracy_score(yb, RandomForestClassifier(100, max_depth=8, random_state=0).fit(xa, ya).predict(xb))],
    boost=[accuracy_score(yb, HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=4, max_leaf_nodes=None, early_stopping=False, random_state=0).fit(xa, ya).predict(xb))])
x, y = datasets.make_friedman1(1000, 8, noise=0.5, random_state=9)
x, y = x.astype(np.float32), y.astype(np.float32)
xa, ya, xb, yb = x[:800], y[:800], x[800:], y[800:]
put("trees_regression", x=xa, y=ya, xt=xb, yt=yb,
    tree=[r2_score(yb, DecisionTreeRegressor(max_depth=6, random_state=0).fit(xa, ya).predict(xb))],
    forest=[r2_score(yb, RandomForestRegressor(100, max_depth=8, random_state=0).fit(xa, ya).predict(xb))],
    boost=[r2_score(yb, HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=4, max_leaf_nodes=None, early_stopping=False, random_state=0).fit(xa, ya).predict(xb))])

# t-SNE: trustworthiness of scikit-learn's exact embedding
x, y = blobs(150, 10, 3, std=2.0, seed=10)
emb = TSNE(2, perplexity=20, method="exact", init="pca", random_state=0, max_iter=1000).fit_transform(x)
put("tsne", x=x, y=y, trust=[trustworthiness(x, emb, n_neighbors=5)])

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
offset = 0
with open(os.path.join(out, "manifest.txt"), "w") as m, open(os.path.join(out, "data.bin"), "wb") as d:
    for case, name, a in arrays:
        kind = "i32" if a.dtype == np.int32 else "f32"
        m.write(f"{case} {name} {kind} {offset} {a.size} {' '.join(map(str, a.shape))}\n")
        b = a.astype("<i4" if kind == "i32" else "<f4").tobytes()
        d.write(b)
        offset += len(b)
print(len(arrays), "arrays,", offset, "bytes")
