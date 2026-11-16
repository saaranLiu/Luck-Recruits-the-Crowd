"""Round-4 figures for the four-page paper.  Reads only round4/results (r4_ledger.csv, r4_curves.csv) and, for the
out-of-sample prediction panel, the round-1/round-3 prediction results already reported in the paper.

fig1_design.pdf    single column  design: what is fixed at the first order, when the outcome clock starts
fig2_random.pdf    single column  (a) calibration of the first outcome by price, (b) out-of-sample prediction gains
fig3_dynamic.pdf   double column  (a,b) standardized return curves, (c) effect over horizon, (d) activity by period
fig4_bounds.pdf    single column  (a) early vs late period, (b) later bets among returners (equivalence band)
"""
import os, sys
R4 = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(R4)
sys.path.insert(0, PROJ)
import numpy as np, pandas as pd
from style import plt                      # noqa: E402  (ACM-matched font, light axes)
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as mt

from matplotlib import font_manager
for _f in ('R', 'B', 'I'):
    font_manager.fontManager.addfont(f'{R4}/fonts/LinBiolinumTT-{_f}.ttf')
TTF = 'Linux Biolinum TT'          # TrueType outlines, so pdf.fonttype 42 embeds a genuine TrueType font
plt.rcParams.update({'font.sans-serif': [TTF, 'DejaVu Sans'], 'mathtext.rm': TTF, 'mathtext.it': f'{TTF}:italic',
                     'mathtext.bf': f'{TTF}:bold'})

FIG = f'{R4}/figures'
os.makedirs(FIG, exist_ok=True)
L = pd.read_csv(f'{R4}/results/r4_ledger.csv').set_index('result_id')
CUR = pd.read_csv(f'{R4}/results/r4_curves.csv')

CR, SP = '#2E6BB5', '#D9438B'
CR_T, SP_T = '#D3E1F3', '#F9D5E6'
GREY, INK, MUTE = '#8C8C8C', '#262626', '#5E5E5E'
MK = {'UD_A': 'o', 'SP_S1': 'D'}
COL = {'UD_A': CR, 'SP_S1': SP}
TINT = {'UD_A': CR_T, 'SP_S1': SP_T}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.titlesize': 8.2, 'legend.fontsize': 7.2, 'lines.solid_capstyle': 'round'})
FA = 7.0          # annotation size


def ptitle(ax, letter, text, x=0.0):
    ax.set_title(f'({letter}) {text}', loc='left', fontweight='bold', pad=3.5, x=x, color=INK)


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png', dpi=300)
    plt.close(fig)


def g(rid, col='estimate_pp'):
    return float(L.loc[rid, col])


# ---- Figure 1: design --------------------------------------------------------------------------------------------------
def fig1():
    fig = plt.figure(figsize=(3.33, 1.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis('off')
    rows = [('UD_A', 37.5, 'Crypto up/down', '(BTC, ETH, SOL, XRP; 5 min to 4 h)', 'window\nopens', 'window\ncloses',
             f'median {g("S:UD_A", "median_lead"):.0f} min'),
            ('SP_S1', 11.0, 'Sports games', '(NBA, NHL, NFL; 94% moneyline)', 'scheduled\nstart', 'game\nends',
             f'median {g("S:SP_S1", "median_lead"):.1f} h')]
    X0, X1, X2, XR, XE = 10, 25, 39.5, 53, 97.5
    for k, y, name, sub, l1, l2, lead in rows:
        c = COL[k]
        t = ax.text(1.0, y + 13.6, name, color=c, fontsize=8.0, fontweight='bold', va='center')
        ax.annotate(sub, xy=(1, 0.5), xycoords=t, xytext=(3, 0), textcoords='offset points', color=MUTE, fontsize=7.0, va='center')
        ax.add_patch(Rectangle((X0, y - 2.5), XR - X0, 5.0, fc='#EDEDED', ec='none', zorder=0))
        ax.add_patch(Rectangle((XR, y - 2.5), XE - XR, 5.0, fc=TINT[k], ec='none', zorder=0, alpha=0.8))
        ax.annotate('', xy=(XE + 2.0, y), xytext=(1.0, y), arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, mutation_scale=6))
        ax.plot(X0, y, marker=MK[k], ms=6.4 if k == 'UD_A' else 5.6, color=c, mec='white', mew=0.7, zorder=6)
        ax.text(X0, y + 3.3, 'first order\n= sample fixed', ha='center', va='bottom', fontsize=FA, fontweight='bold', color=c,
                linespacing=0.95)
        for x, lab in [(X1, l1), (X2, l2)]:
            ax.plot([x, x], [y - 1.9, y + 1.9], color=INK, lw=0.8, zorder=5)
            ax.text(x, y + 3.4, lab, ha='center', va='bottom', fontsize=FA, color=MUTE, linespacing=0.95)
        ax.annotate('', xy=(X1, y - 4.2), xytext=(X0, y - 4.2),
                    arrowprops=dict(arrowstyle='<->', lw=0.5, color=MUTE, mutation_scale=4, shrinkA=0, shrinkB=0))
        ax.text((X0 + X1) / 2, y - 5.0, lead, ha='center', va='top', fontsize=6.8, color=MUTE)
        ax.text((X1 + XR) / 2 + 7.5, y - 3.4, 'later sells, hedges, trades:\noutcomes, not filters', ha='center', va='top',
                fontsize=6.6, color=MUTE, style='italic', linespacing=0.95)
        ax.plot(XR, y, marker='o', ms=4.2, color=INK, zorder=6)
        ax.text(XR, y + 3.4, 'resolution\n= day 0', ha='center', va='bottom', fontsize=FA, color=INK, linespacing=0.95)
        for x, lab in [(65, '7 d'), (79, '30 d'), (93, '90 d')]:
            ax.plot([x, x], [y - 1.5, y + 1.5], color=c, lw=0.8, zorder=5)
            ax.text(x, y + 3.4, lab, ha='center', va='bottom', fontsize=FA, color=c)
        ax.text((XR + XE) / 2 + 1, y - 3.6, 'return and activity', ha='center', va='top', fontsize=6.8, color=c)
    save(fig, 'fig1_design')


# ---- Figure 2: is the first outcome predictable? -------------------------------------------------------------------------
def fig2():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.33, 1.95), gridspec_kw=dict(width_ratios=[1.0, 1.1], wspace=0.1,
                                                                               left=0.125, right=0.99, bottom=0.19, top=0.88))
    a1.plot([28, 74], [28, 74], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    a1.text(30.5, 76.5, '- - won = price', fontsize=6.6, color=MUTE, ha='left', va='center')
    for k in ['SP_S1', 'UD_A']:
        rows = L[L.index.str.startswith(f'K:{k}:') & (L.n_wallets >= 200)]
        sz = 9 * np.sqrt(rows.n_wallets / 1000)
        a1.errorbar(rows.mean_price_c, rows.estimate_pp, yerr=[rows.estimate_pp - rows.ci_low_pp, rows.ci_high_pp - rows.estimate_pp],
                    fmt='none', ecolor=COL[k], elinewidth=0.7, alpha=0.85, zorder=2)
        a1.scatter(rows.mean_price_c, rows.estimate_pp, s=sz, marker=MK[k], color=COL[k], edgecolor='white', lw=0.5, zorder=3,
                   alpha=0.95)
    a1.plot(31.6, 68.5, marker='D', ms=3.6, color=SP, mec='white', mew=0.4)
    a1.text(33.4, 68.5, 'sports bins', fontsize=6.9, color=SP, fontweight='bold', va='center')
    a1.plot(31.6, 61.5, marker='o', ms=4.0, color=CR, mec='white', mew=0.4)
    a1.text(33.4, 61.5, 'crypto bins', fontsize=6.9, color=CR, fontweight='bold', va='center')
    a1.set_xlim(29, 73); a1.set_ylim(15, 82)
    a1.set_xticks([30, 40, 50, 60, 70]); a1.set_yticks([20, 40, 60, 80])
    a1.set_xlabel('Price paid (cents)', labelpad=1.5); a1.set_ylabel('First side won (%)', labelpad=1.5)
    a1.yaxis.grid(True, color='0.92', lw=0.5); a1.set_axisbelow(True)
    ptitle(a1, 'a', 'Win rate by price')

    p = pd.read_csv(f'{PROJ}/results/rand_prediction.csv')
    p = p[(p['sample'] == 'UD_A') & (p.cluster == 'cid')].set_index('model')
    q = pd.read_csv(f'{PROJ}/round3/pipeline/results/R_sp_prediction.csv')
    q = q[q['sample'] == 'SP_S1'].set_index('model')
    items = [('UD_A', 'crypto: pre-order, logistic', p.loc['p_logit'], 4.85), ('UD_A', 'crypto: pre-order, trees', p.loc['p_gbm'], 3.95),
             ('SP_S1', 'sports: price only', q.loc['p_base'], 3.05),
             ('SP_S1', 'sports: + pre-order, logistic', q.loc['p_logit'], 1.15), ('SP_S1', 'sports: + pre-order, trees', q.loc['p_gbm'], 0.25)]
    for k, lab, r, y in items:
        a2.plot([r.ll_lo, r.ll_hi], [y, y], color=COL[k], lw=1.0, solid_capstyle='butt')
        a2.plot(r.ll_gain_pct, y, marker=MK[k], ms=3.6 if k == 'UD_A' else 3.2, color=COL[k], mec='white', mew=0.4, zorder=3)
        a2.text(-10.3, y + 0.22, lab, fontsize=6.5, color=COL[k], va='bottom', ha='left')
    a2.axvline(0, color=INK, lw=0.6)
    a2.text(-10.3, 5.7, 'Gain over constant', fontsize=6.9, color=INK, fontweight='bold', va='bottom')
    a2.axhline(2.62, xmin=0.0, xmax=1.0, color='0.75', lw=0.5, ls=(0, (2, 2)))
    a2.text(-10.3, 1.98, 'Incremental gain over price', fontsize=6.9, color=INK, fontweight='bold', va='bottom', zorder=5,
            bbox=dict(fc='white', ec='none', pad=0.6))
    a2.set_ylim(-0.35, 6.45); a2.set_yticks([]); a2.spines['left'].set_visible(False)
    a2.set_xlim(-10.5, 6.5); a2.set_xticks([-8, -4, 0, 4])
    a2.set_xlabel('Log-loss gain (%)', labelpad=1.5)
    a2.text(6.3, -0.3, 'better', fontsize=6.6, color=MUTE, ha='right', va='bottom', style='italic')
    ptitle(a2, 'b', 'Out-of-sample gain', x=0.02)
    save(fig, 'fig2_random')


# ---- Figure 3: returning versus staying --------------------------------------------------------------------------------
HT = [1, 24, 168, 720, 2160]
HL = ['1h', '1d', '7d', '30d', '90d']


def logx(ax):
    ax.set_xscale('log'); ax.set_xlim(0.4, 2600)
    ax.set_xticks(HT); ax.set_xticklabels(HL)
    ax.xaxis.set_minor_locator(mt.NullLocator())
    tl = ax.get_xticklabels()
    tl[4].set_ha('left')
    for h in [168, 720, 2160]:
        ax.axvline(h, color='0.9', lw=0.5, zorder=0)


def fig3():
    fig = plt.figure(figsize=(7.0, 2.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.08, 1.2], wspace=0.36, left=0.055, right=0.995, bottom=0.205, top=0.88)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax, k, nm, letter in [(axs[0], 'UD_A', 'Crypto', 'a'), (axs[1], 'SP_S1', 'Sports', 'b')]:
        c = CUR[(CUR['sample'] == k) & (CUR.hours >= 0.4)].sort_values('hours')
        ax.fill_between(c.hours, c.lost_std, c.won_std, color=TINT[k], lw=0, zorder=1)
        ax.plot(c.hours, c.lost_std, color=GREY, lw=1.1, zorder=2)
        ax.plot(c.hours, c.won_std, color=COL[k], lw=1.3, zorder=3)
        logx(ax)
        ax.set_ylim(0, 92); ax.set_yticks([0, 20, 40, 60, 80])
        for h, dy in [(168, 0), (2160, 0)]:
            r = c[c.hours == h].iloc[0]
            ax.plot([h, h], [r.lost_std, r.won_std], color=COL[k], lw=0.6, zorder=4)
            ax.plot(h, r.won_std, marker=MK[k], ms=3.0, color=COL[k], mec='white', mew=0.4, zorder=5)
            ax.plot(h, r.lost_std, marker=MK[k], ms=3.0, color='white', mec=GREY, mew=0.7, zorder=5)
            ax.text(h * (0.80 if h == 2160 else 0.78), r.won_std + 4.0, f'+{r.b:.1f}', ha='center', va='bottom', fontsize=FA,
                    color=COL[k], fontweight='bold')
        i0 = c.index[np.argmin(np.abs(c.hours - 6.75))]
        ax.text(1.0, 62, 'won', color=COL[k], fontsize=7.4, fontweight='bold')
        ax.text(1.0, 53, 'lost', color=GREY, fontsize=7.4, fontweight='bold')
        if k == 'UD_A':
            ax.set_ylabel('Traded again by time $t$ (%)', labelpad=1.5)
        ax.set_xlabel('Time since resolution, $t$ (log scale)', labelpad=1.5)
        ptitle(ax, letter, f'{nm}: return curves')
    ax = axs[2]
    for k in ['UD_A', 'SP_S1']:
        c = CUR[(CUR['sample'] == k) & (CUR.hours >= 0.4)].sort_values('hours')
        ax.fill_between(c.hours, c.lo, c.hi, color=COL[k], alpha=0.16, lw=0, zorder=1)
        ax.plot(c.hours, c.b, color=COL[k], lw=1.2, zorder=3)
        m = c[c.hours.isin([168, 720, 2160])]
        ax.plot(m.hours, m.b, ls='none', marker=MK[k], ms=3.2, color=COL[k], mec='white', mew=0.4, zorder=4)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    logx(ax); ax.set_ylim(-4.4, 7.2); ax.set_yticks([-4, -2, 0, 2, 4, 6])
    ax.set_ylabel('Won minus lost (pp)', labelpad=1.0)
    ax.set_xlabel('Time since resolution, $t$ (log scale)', labelpad=1.5)
    dcr, dsp = L.loc['D:UD_A:ret30d-ret7d'], L.loc['D:SP_S1:ret30d-ret7d']
    ax.text(0.5, 3.75, 'Crypto', color=CR, fontsize=7.2, fontweight='bold')
    ax.text(30, 6.25, 'Sports', color=SP, fontsize=7.2, fontweight='bold', ha='center')
    ax.text(0.5, -1.15, '7d to 30d change:', color=INK, fontsize=6.6, va='top')
    ax.text(0.5, -2.2, f'crypto {dcr.estimate_pp:.1f} [{dcr.ci_low_pp:.1f}, {dcr.ci_high_pp:.1f}]', color=CR, fontsize=6.6, va='top')
    ax.text(0.5, -3.2, f'sports {dsp.estimate_pp:.1f} [{dsp.ci_low_pp:.1f}, {dsp.ci_high_pp:.1f}]', color=SP, fontsize=6.6, va='top')
    ptitle(ax, 'c', 'Effect on cumulative return')
    ax = axs[3]
    per = ['act0_7', 'act8_14', 'act15_21', 'act22_30', 'act31_60', 'act61_90']
    lab = ['1-7', '8-14', '15-21', '22-30', '31-60', '61-90']
    for k, off in [('UD_A', -0.16), ('SP_S1', 0.16)]:
        r = L.loc[[f'P:{k}:{p}' for p in per]]
        x = np.arange(len(per)) + off
        ax.vlines(x, r.ci_low_pp, r.ci_high_pp, color=COL[k], lw=1.0)
        ax.plot(x, r.estimate_pp, ls='none', marker=MK[k], ms=3.4 if k == 'UD_A' else 3.0, color=COL[k], mec='white', mew=0.4, zorder=3)
    ax.axhline(0, color=INK, lw=0.6)
    for i in range(len(per)):
        if i % 2 == 1:
            ax.axvspan(i - 0.5, i + 0.5, color='0.965', lw=0, zorder=0)
    ax.set_xlim(-0.55, len(per) - 0.45)
    ax.set_xticks(range(len(per))); ax.set_xticklabels(lab, rotation=30, ha='right', rotation_mode='anchor', fontsize=7)
    ax.set_ylim(-4.4, 7.2); ax.set_yticks([-4, -2, 0, 2, 4, 6])
    ax.set_xlabel('Window (days since resolution)', labelpad=1.0)
    ax.set_ylabel('Won minus lost (pp)', labelpad=1.0)
    ax.legend(handles=[Line2D([], [], color=CR, marker='o', ms=3.2, lw=1.0, label='Crypto'),
                       Line2D([], [], color=SP, marker='D', ms=2.8, lw=1.0, label='Sports')],
              loc='lower left', ncol=2, handlelength=1.1, columnspacing=0.8, borderaxespad=0.1, handletextpad=0.4)
    ptitle(ax, 'd', 'Activity in each window')
    save(fig, 'fig3_dynamic')


# ---- Figure 4: boundaries ---------------------------------------------------------------------------------------------
def fig4():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.33, 1.92), gridspec_kw=dict(width_ratios=[1.3, 1.0], wspace=0.45,
                                                                              left=0.215, right=0.985, bottom=0.185, top=0.825))
    rows = [('UD_A', 'ret7d', 'Crypto, 1 wk'), ('UD_A', 'act61_90', 'Crypto, d61-90'),
            ('SP_S1', 'ret7d', 'Sports, 1 wk'), ('SP_S1', 'act61_90', 'Sports, d61-90')]
    ys = [3.45, 2.35, 1.1, 0.0]
    for (k, y, lab), yy in zip(rows, ys):
        e, l_ = L.loc[f'E:{k}:{y}:early'], L.loc[f'E:{k}:{y}:late']
        for r, dy, filled in [(e, 0.19, False), (l_, -0.19, True)]:
            a1.plot([r.ci_low_pp, r.ci_high_pp], [yy + dy, yy + dy], color=COL[k], lw=0.9, alpha=1 if filled else 0.6, solid_capstyle='butt')
            a1.plot(r.estimate_pp, yy + dy, marker=MK[k], ms=3.4 if k == 'UD_A' else 3.0, color=COL[k] if filled else 'white',
                    mec=COL[k], mew=0.8, zorder=3)
    a1.set_yticks(ys); a1.set_yticklabels([r[2] for r in rows], fontsize=7.0)
    for t, (k, _, _) in zip(a1.get_yticklabels(), rows):
        t.set_color(COL[k])
    a1.tick_params(axis='y', length=0, pad=1.5)
    dd = L.loc['E:SP_S1:act61_90:late minus early']
    a1.text(12.1, ys[3], f'late \u2212 early:\n\u2212{abs(dd.estimate_pp):.1f} (SE {dd.se_2way:.1f})', fontsize=6.5, color=SP,
            ha='right', va='center', linespacing=0.95)
    a1.axvline(0, color=INK, lw=0.6)
    a1.set_xlim(-5.2, 12.2); a1.set_xticks([-4, 0, 4, 8]); a1.set_ylim(-0.5, 4.35)
    a1.spines['left'].set_visible(False)
    a1.set_xlabel('Won minus lost (pp)', labelpad=1.5)
    a1.legend(handles=[Line2D([], [], ls='none', marker='o', ms=3.2, mfc='white', mec=INK, mew=0.8, label='early'),
                       Line2D([], [], ls='none', marker='o', ms=3.2, color=INK, label='late')],
              loc='upper right', ncol=2, handletextpad=0.05, columnspacing=0.5, borderaxespad=0.0, bbox_to_anchor=(1.02, 1.03),
              fontsize=6.8)
    ptitle(a1, 'a', 'Effect by period', x=-0.45)

    a2.axvspan(-1, 1, color='#E9E9E9', lw=0, zorder=0)
    a2.text(-1.12, 2.05, '±1 cent', fontsize=6.5, color=MUTE, ha='right', va='bottom')
    for k, yy in [('UD_A', 1.25), ('SP_S1', 0.25)]:
        r = L.loc[f'Q:{k}:edge_share']
        a2.plot([r.ci_low_pp, r.ci_high_pp], [yy, yy], color=COL[k], lw=1.1, solid_capstyle='butt')
        a2.plot(r.estimate_pp, yy, marker=MK[k], ms=3.6 if k == 'UD_A' else 3.2, color=COL[k], mec='white', mew=0.4, zorder=3)
        a2.text(-3.15, yy - 0.18, f'TOST p = {r.p_2way:.3f}'.replace('0.', '.'), fontsize=6.5, color=MUTE, va='top', ha='left')
    a2.set_yticks([1.25, 0.25]); a2.set_yticklabels(['Crypto', 'Sports'], fontsize=7.0)
    for t, k in zip(a2.get_yticklabels(), ['UD_A', 'SP_S1']):
        t.set_color(COL[k])
    a2.tick_params(axis='y', length=0, pad=1.5)
    a2.axvline(0, color=INK, lw=0.6)
    a2.set_xlim(-3.2, 1.6); a2.set_xticks([-3, -2, -1, 0, 1]); a2.set_ylim(-0.45, 2.5)
    a2.spines['left'].set_visible(False)
    a2.set_xlabel('Difference (cents)', labelpad=1.5)
    ptitle(a2, 'b', 'Later trading edge\namong returners', x=-0.3)
    save(fig, 'fig4_bounds')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print('figures written to', FIG)
