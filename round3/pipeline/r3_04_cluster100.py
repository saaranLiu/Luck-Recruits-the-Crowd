"""Round-3 rerun of the up/down $95-105 one-shot cluster (21_hetero_ex.py), whose model used the ex-post timing control
(ldelay).  Only the timing control changes (lsched); rows and clustering are those of the round-1 script."""
from r2_common import *
import numpy as np, pandas as pd
from r2_samples import load_r2, samples_r2, spec, ols2

SP_OLD = dict(ctrl=('lstake', 'ldelay'), cat=('pbin', 'cell', 'hour'))
d = load_r2()
S = samples_r2(d)
rows = []
for k in ['UD_A', 'UD_C']:
    v = S[k]
    c = v.stake0.between(95, 105)
    for lab, g in [('stake 95-105', v[c]), ('other', v[~c])]:
        for tl, sp in [('M1 scheduled end (round 3)', spec(k)), ('M0 actual resolution (round 2)', SP_OLD)]:
            for y in ['ret7d', 'ret30d']:
                r = ols2(g, y, **sp)
                rows.append(dict(sample=k, level=lab, timing=tl, share=len(g) / len(v),
                                 ever_again=(g.n_all.fillna(0) > 0).mean(), **r))
out = pd.DataFrame(rows)
out.to_csv(f'{R2RES}/R3_cluster100.csv', index=False)
pd.set_option('display.width', 250)
print(out[['sample', 'level', 'timing', 'y', 'b', 'se', 'p', 'n', 'base', 'share', 'ever_again']].round(4).to_string())
log('done')
