"""Table 1 of the four-page paper (samples, one-week effect, robustness), written from round4/results/r4_ledger.csv only."""
import os
import pandas as pd

R4 = os.path.dirname(os.path.abspath(__file__))
L = pd.read_csv(f'{R4}/results/r4_ledger.csv').set_index('result_id')
TAB = f'{R4}/../results/tables'
os.makedirs(TAB, exist_ok=True)
OUT = f'{TAB}/tab1_main.tex'


def ci(rid, d=2):
    r = L.loc[rid]
    return f'{r.estimate_pp:.{d}f} [{r.ci_low_pp:.{d}f}, {r.ci_high_pp:.{d}f}]'


def bci(rid):
    r = L.loc[rid]
    return f'[{r.ci_low_pp:.2f}, {r.ci_high_pp:.2f}]'


def row(label, a, b):
    return f'{label} & {a} & {b} \\\\'


sa, ss = L.loc['S:UD_A'], L.loc['S:SP_S1']
ma, ms = L.loc['M:UD_A:ret7d'], L.loc['M:SP_S1:ret7d']
rows = [
    r'\multicolumn{3}{l}{\emph{Sample, fixed at the first order}} \\',
    row('Newcomer wallets', f'{int(sa.n_wallets):,}', f'{int(ss.n_wallets):,}'),
    row('Markets / games', f'{int(sa.n_markets):,} / --', f'{int(ss.n_markets):,} / {int(ss.n_games):,}'),
    row('First orders', 'Oct 2025--Apr 2026', 'Feb 2023--Apr 2026'),
    row('Median stake / lead', f'\\${sa.median_stake:.2f} / {sa.median_lead:.0f} min', f'\\${ss.median_stake:.2f} / {ss.median_lead:.1f} h'),
    row('First side won (\\%)', f'{100 * sa.win_share:.1f}', f'{100 * ss.win_share:.1f}'),
    row('Losers back within 1 week (\\%)', f'{ma.base_lost_raw:.1f}', f'{ms.base_lost_raw:.1f}'),
    r'\midrule',
    r'\multicolumn{3}{l}{\emph{Effect of a first win on return within 1 week (pp) [95\% CI]}} \\',
    row('Main specification', ci('M:UD_A:ret7d'), ci('M:SP_S1:ret7d')),
    row('\\quad same market (market FE)', ci('R:UD_A:market_fe'), ci('R:SP_S1:market_fe')),
    row('\\quad rich pre-order controls$^a$', ci('R:UD_A:rich'), ci('R:SP_S1:A2_SPR_M1')),
    row('\\quad timing control$^b$', f'{L.loc["R:UD_A:A2_UD_M0", "estimate_pp"]:.2f}--{L.loc["R:UD_A:A2_UD_M2", "estimate_pp"]:.2f}',
        ci('R:SP_S1:A2_SPM_M1')),
    row('\\quad order $\\geq$1 h before start', '--', ci('R:SP_S1:lead1h')),
    row('\\quad without flagged NBA orders', '--', ci('R:SP_S1:excl_nba_flags')),
    row('\\quad bootstrap: days', bci('R:UD_A:boot_rday'), bci('R:SP_S1:boot_rday')),
    row('\\quad bootstrap: weeks', bci('R:UD_A:boot_rweek'), bci('R:SP_S1:boot_rweek')),
    row('\\quad placebo / simulated null $p^c$', f'{L.loc["R:UD_A:placebo", "p_2way"]:.3f}', f'{L.loc["R:SP_S1:simnull", "p_2way"]:.4f}'),
]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(rows) + '\n')
print(open(OUT).read())

OUT2 = f'{TAB}/tab2_dynamic.tex'


def num(x, d):
    return f'{x:.{d}f}'.replace('-', '$-$')


def ci1(rid, d=1):
    r = L.loc[rid]
    return f'{num(r.estimate_pp, d)} [{num(r.ci_low_pp, d)}, {num(r.ci_high_pp, d)}]'


def line(label, a, s, base=True, d=1):
    la = num(L.loc[a].base_lost_raw, 1) if base else ''
    ls = num(L.loc[s].base_lost_raw, 1) if base else ''
    return f'{label} & {la} & {ci1(a, d)} & {ls} & {ci1(s, d)} \\\\'


ca, cs = L.loc['C:UD_A:2160h'], L.loc['C:SP_S1:2160h']
rows2 = [
    r'\multicolumn{5}{@{}l}{\emph{Traded again by}} \\',
    line('\\quad 7 days', 'C:UD_A:168h', 'C:SP_S1:168h'),
    line('\\quad 30 days', 'C:UD_A:720h', 'C:SP_S1:720h'),
    line('\\quad 90 days', 'C:UD_A:2160h', 'C:SP_S1:2160h'),
    line('\\quad change, 7 to 30 days$^a$', 'D:UD_A:ret30d-ret7d', 'D:SP_S1:ret30d-ret7d', base=False),
    line('\\quad change, 7 to 90 days$^a$', 'D:UD_A:ret90d-ret7d', 'D:SP_S1:ret90d-ret7d', base=False),
    r'\midrule',
    r'\multicolumn{5}{@{}l}{\emph{Active in}} \\',
    line('\\quad days 8--14', 'P:UD_A:act8_14', 'P:SP_S1:act8_14'),
    line('\\quad days 22--30', 'P:UD_A:act22_30', 'P:SP_S1:act22_30'),
    line('\\quad days 61--90', 'P:UD_A:act61_90', 'P:SP_S1:act61_90'),
    line('\\quad no.\\ of days, 1--30$^b$', 'P:UD_A:days30', 'P:SP_S1:days30', d=2),
    line('\\quad no.\\ of weeks, 1--13$^b$', 'P:UD_A:weeks90', 'P:SP_S1:weeks90', d=2),
]
open(OUT2, 'w').write('\n'.join(rows2) + '\n')
print(open(OUT2).read())
