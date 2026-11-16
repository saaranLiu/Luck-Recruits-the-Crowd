"""Shared figure style: font matched to the ACM template, a blue/pink sample palette with light-green accents, light grids."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = 'Linux Biolinum O'
if FONT not in {f.name for f in font_manager.fontManager.ttflist}:
    FONT = 'Lato'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'DejaVu Sans'], 'font.size': 8,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic',
    'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'axes.edgecolor': '0.25', 'axes.labelcolor': '0.15', 'xtick.color': '0.25', 'ytick.color': '0.25',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.grid': False, 'grid.color': '0.91', 'grid.linewidth': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})

COL = {'UD_pre': '#3A6EB5', 'SP_pre': '#D65A8C'}          # blue, pink
TINT = {'UD_pre': '#D6E3F4', 'SP_pre': '#F9D9E6'}          # light versions for fills
NAME = {'UD_pre': 'Crypto up/down', 'SP_pre': 'Sports moneyline'}
SHORT = {'UD_pre': 'Up/down', 'SP_pre': 'Sports'}
LOSE = '#8A8A8A'
GREEN, GREEN_T, GREEN_L = '#4E9B6C', '#CBE7D3', '#EAF5ED'   # neutral accents
INK = '0.2'


def ygrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)


def title(ax, letter, text, **kw):
    ax.set_title(f'({letter}) {text}', loc='left', fontweight='bold', pad=4, **kw)
