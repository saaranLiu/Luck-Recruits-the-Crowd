"""Fast OLS with dummy fixed effects, cluster-robust (one- or two-way) SEs, and block bootstraps, used by T5-T7."""
import numpy as np, pandas as pd


def build(s, y, x, ctrl, cat, inter=None):
    """Design matrix [x, (x*inter), ctrl, dummies(cat)] + constant; rows with missing values must be dropped first."""
    cols = [s[x].astype(float).to_numpy()[:, None]]
    names = [x]
    if inter is not None:
        cols.append((s[x].astype(float) * s[inter].astype(float)).to_numpy()[:, None]); names.append(f'{x}:{inter}')
    for c in ctrl:
        cols.append(s[c].astype(float).to_numpy()[:, None]); names.append(c)
    for c in cat:
        dm = pd.get_dummies(s[c].astype(str), prefix=c, drop_first=True, dtype=float)
        cols.append(dm.to_numpy()); names += list(dm.columns)
    cols.append(np.ones((len(s), 1))); names.append('const')
    return np.hstack(cols), s[y].astype(float).to_numpy(), names


def ols_fit(X, Y):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    return beta, XtX_inv, Y - X @ beta


def cluster_V(X, e, XtX_inv, g):
    g = pd.factorize(g)[0]
    G = g.max() + 1
    sc = np.zeros((G, X.shape[1]))
    np.add.at(sc, g, X * e[:, None])
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k)
    return adj * XtX_inv @ (sc.T @ sc) @ XtX_inv, G


def cluster_se(X, e, XtX_inv, groups, j=0):
    """groups: one array (one-way) or a list of two arrays (two-way, Cameron-Gelbach-Miller)."""
    if isinstance(groups, list) and len(groups) == 2:
        V1, G1 = cluster_V(X, e, XtX_inv, groups[0]); V2, G2 = cluster_V(X, e, XtX_inv, groups[1])
        inter = pd.factorize(pd.Series(groups[0]).astype(str) + '|' + pd.Series(groups[1]).astype(str))[0]
        V12, _ = cluster_V(X, e, XtX_inv, inter)
        V = V1 + V2 - V12
        return float(np.sqrt(max(V[j, j], 0))), min(G1, G2)
    V, G = cluster_V(X, e, XtX_inv, groups)
    return float(np.sqrt(V[j, j])), G


def block_boot(X, Y, blocks, B, rng, j=0):
    b = pd.factorize(blocks)[0]
    M = b.max() + 1
    order = np.argsort(b, kind='stable'); bounds = np.searchsorted(b[order], np.arange(M + 1))
    out = np.empty(B)
    for i in range(B):
        draw = rng.integers(0, M, M)
        ix = np.concatenate([order[bounds[m]:bounds[m + 1]] for m in draw])
        Xi = X[ix]
        keep = Xi.std(axis=0) > 0
        keep[-1] = True
        beta, _, _ = ols_fit(Xi[:, keep], Y[ix])
        out[i] = beta[np.flatnonzero(keep).tolist().index(j)]
    return out
