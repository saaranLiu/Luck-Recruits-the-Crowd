"""Design matrix and a clustered OLS used by the sample code."""
import pandas as pd
import statsmodels.api as sm


def design(s, x, ctrl, cat, extra=()):
    return pd.concat([s[[x] + list(ctrl) + list(extra)].astype(float)] +
                     [pd.get_dummies(s[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in cat], axis=1)


def ols(df, y, x='win', ctrl=('lstake',), cat=('pbin', 'month', 'hour'), fe=None, extra=()):
    """OLS of y on x and controls. SE clustered by market."""
    cols = [y, x, 'cid'] + list(ctrl) + list(cat) + list(extra) + ([fe] if fe else [])
    s = df[list(dict.fromkeys(cols))].dropna()
    if fe:
        s = s[s.groupby(fe)[x].transform('nunique') > 1]
    if len(s) < 200:
        return None
    X = design(s, x, ctrl, cat, extra)
    Y = s[y].astype(float)
    if fe:
        X = X - X.groupby(s[fe]).transform('mean')
        Y = Y - Y.groupby(s[fe]).transform('mean')
        X = X.loc[:, X.std() > 1e-12]
    else:
        X = sm.add_constant(X)
    r = sm.OLS(Y, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s['cid'])[0]})
    return dict(y=y, b=r.params[x], se=r.bse[x], p=r.pvalues[x], n=len(s), n_mkt=s.cid.nunique(),
                base=s.loc[s[x] == 0, y].mean(), win_rate=s[x].mean())
