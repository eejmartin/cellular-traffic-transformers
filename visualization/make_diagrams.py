# -*- coding: utf-8 -*-
"""Schematic diagrams (Macedonian labels) for the thesis.

Self-contained — no run data needed. Output dir is set in the CONFIG block.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

# --- CONFIG -----------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "master_thesis", "src", "figures")   # output folder
# ----------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; AQUA = "#1baf7a"; YELLOW = "#eda100"; GREEN = "#008300"
VIOLET = "#4a3aa7"; RED = "#e34948"; ORANGE = "#eb6834"
LIGHTBLUE = "#cde2fb"; MIDBLUE = "#86b6ef"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def box(ax, x, y, w, h, text, fc="#f0f4fb", ec=BLUE, fs=9, lw=1.4, tc=INK, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.4, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                        color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---------------------------------------------------------------- 1. mobile arch
fig, ax = plt.subplots(figsize=(9.5, 3.4)); blank(ax)
box(ax, 0.02, 0.32, 0.15, 0.36, "Кориснички\nуреди (UE)", fc="#eef7f2", ec=AQUA, fs=10)
box(ax, 0.24, 0.16, 0.26, 0.68, "Радио пристапна мрежа\n(RAN)\n\nБазни станици\n(eNB / gNB)", fc="#f0f4fb", ec=BLUE, fs=10)
box(ax, 0.57, 0.16, 0.22, 0.68, "Јадрена мрежа\n(Core / EPC / 5GC)\n\nУправување со сесии,\nмобилност, наплата", fc="#fdf4ec", ec=ORANGE, fs=9.5)
box(ax, 0.86, 0.32, 0.12, 0.36, "Интернет /\nуслуги", fc="#f4f2fb", ec=VIOLET, fs=10)
arrow(ax, 0.17, 0.5, 0.24, 0.5); arrow(ax, 0.24, 0.44, 0.17, 0.44)
arrow(ax, 0.50, 0.5, 0.57, 0.5); arrow(ax, 0.57, 0.44, 0.50, 0.44)
arrow(ax, 0.79, 0.5, 0.86, 0.5); arrow(ax, 0.86, 0.44, 0.79, 0.44)
ax.text(0.205, 0.56, "радио\nинтерфejс", ha="center", fontsize=8, color=MUTED)
ax.text(0.535, 0.56, "backhaul", ha="center", fontsize=8, color=MUTED)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_arch_mobile.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 2. O-RAN
fig, ax = plt.subplots(figsize=(9.5, 5.4)); blank(ax)
box(ax, 0.03, 0.80, 0.94, 0.17, "SMO (Service Management and Orchestration)\nNon-RT RIC  •  rApps  (контролна јамка > 1 s: политики, тренирање на ML модели)",
    fc="#f0f4fb", ec=BLUE, fs=9.5, bold=False)
box(ax, 0.18, 0.52, 0.64, 0.17, "Near-RT RIC  •  xApps\n(контролна јамка 10 ms – 1 s: оптимизација во речиси реално време)",
    fc="#eef7f2", ec=AQUA, fs=9.5)
box(ax, 0.06, 0.24, 0.26, 0.17, "O-CU\n(Central Unit)", fc="#fdf4ec", ec=ORANGE, fs=9.5)
box(ax, 0.38, 0.24, 0.26, 0.17, "O-DU\n(Distributed Unit)", fc="#fdf4ec", ec=ORANGE, fs=9.5)
box(ax, 0.70, 0.24, 0.24, 0.17, "O-RU\n(Radio Unit)", fc="#fdf4ec", ec=ORANGE, fs=9.5)
box(ax, 0.06, 0.015, 0.88, 0.13, "ML модели за предвидување на сообраќај (како во овој труд)\n→ rApps / xApps за предиктивна оптимизација", fc="#fbf7ec", ec=YELLOW, fs=9.5)
arrow(ax, 0.5, 0.80, 0.5, 0.69); ax.text(0.515, 0.745, "A1 (политики)", fontsize=8.5, color=MUTED, ha="left")
arrow(ax, 0.30, 0.52, 0.19, 0.41); ax.text(0.20, 0.475, "E2", fontsize=8.5, color=MUTED)
arrow(ax, 0.50, 0.52, 0.51, 0.41); ax.text(0.52, 0.475, "E2", fontsize=8.5, color=MUTED)
arrow(ax, 0.32, 0.325, 0.38, 0.325); ax.text(0.35, 0.35, "F1", fontsize=8.5, color=MUTED, ha="center")
arrow(ax, 0.64, 0.325, 0.70, 0.325); ax.text(0.67, 0.35, "OFH", fontsize=8.5, color=MUTED, ha="center")
arrow(ax, 0.5, 0.15, 0.5, 0.24, color=MUTED, ls=(0,(4,3)))
arrow(ax, 0.14, 0.15, 0.10, 0.80, color=MUTED, ls=(0,(4,3)))
plt.tight_layout(); plt.savefig(f"{OUT}/fig_oran.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 3. pipeline
fig, ax = plt.subplots(figsize=(10.5, 2.9)); blank(ax)
steps = [
    ("Прибирање\nна податоци", "91 CSV (bs) /\n17k станици (big_data)", AQUA, "#eef7f2"),
    ("Претпроцесирање", "агрегација, прозорци,\nподелба 70/10/20", BLUE, "#f0f4fb"),
    ("Тренирање", "PatchTST, CATS,\nDeformableTST,\nPeri-midFormer", VIOLET, "#f4f2fb"),
    ("Валидација и\nтестирање", "рано запирање,\nMSE/MAE/RSE", ORANGE, "#fdf4ec"),
    ("Споредбена\nанализа", "по модел, по станица,\nпо категорија", YELLOW, "#fbf7ec"),
    ("Интерпретација\nво RAN контекст", "оптимизација на\nресурси и енергија", GREEN, "#eef6ee"),
]
n = len(steps); w = 0.145; gap = (1 - n*w) / (n - 1) if n > 1 else 0
for i, (t, sub, ec, fc) in enumerate(steps):
    x = i * (w + gap)
    box(ax, x, 0.30, w, 0.52, "", fc=fc, ec=ec)
    ax.text(x + w/2, 0.70, t, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + w/2, 0.45, sub, ha="center", va="center", fontsize=7.8, color=INK2, linespacing=1.4)
    if i < n - 1:
        arrow(ax, x + w, 0.56, x + w + gap, 0.56, lw=1.6)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_pipeline.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 4. windowing
rng = np.random.default_rng(3)
t = np.arange(60)
serie = 30 + 14*np.sin((t-7)/24*2*np.pi) + 5*np.sin(t/12*2*np.pi) + rng.normal(0, 1.8, 60)
fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True)
for k, ax in enumerate(axes):
    start = k * 6
    ax.plot(t, serie, color=BASE, lw=1.6, zorder=1)
    ax.plot(t[start:start+24], serie[start:start+24], color=BLUE, lw=2.4, zorder=3)
    ax.scatter([t[start+24]], [serie[start+24]], color=RED, s=48, zorder=4)
    ax.axvspan(t[start], t[start+23], color=LIGHTBLUE, alpha=0.45, zorder=0)
    ax.axvspan(t[start+23]+0.5, t[start+24]+0.5, color="#fbdddd", zorder=0)
    ax.set_ylabel("корисници", fontsize=9, color=INK2)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.annotate(f"влезен прозорец (seq_len = 24 ч.)", xy=(t[start+11], max(serie)+2),
                ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.annotate("цел (pred_len = 1 ч.)", xy=(t[start+24], serie[start+24]),
                xytext=(t[start+24]+6, serie[start+24]+9), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.set_ylim(min(serie)-4, max(serie)+7)
axes[0].set_title("Лизгачки прозорец: примерок k", fontsize=10, color=INK, loc="left")
axes[1].set_title("Лизгачки прозорец: примерок k+6 (поместен за 6 часа)", fontsize=10, color=INK, loc="left")
axes[1].set_xlabel("време (часови)", fontsize=9, color=INK2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_windowing.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 5. train/val/test split
import pandas as pd
df = pd.read_csv(os.path.join(REPO, "data/big_bs/bs_6260.csv"))
u = df["users"].values; N = len(u)
n_tr, n_va = int(N*0.7), int(N*0.1)
fig, ax = plt.subplots(figsize=(9.5, 3.2))
x = np.arange(N)
ax.plot(x[:n_tr+1], u[:n_tr+1], color=BLUE, lw=1.6, label=f"тренирачко множество (70 %, {n_tr} ч.)")
ax.plot(x[n_tr:n_tr+n_va+1], u[n_tr:n_tr+n_va+1], color=YELLOW, lw=1.8, label=f"валидациско множество (10 %, {n_va} ч.)")
ax.plot(x[n_tr+n_va:], u[n_tr+n_va:], color=RED, lw=1.8, label=f"тест множество (20 %, {N-n_tr-n_va} ч.)")
ax.axvline(n_tr, color=MUTED, lw=0.9, ls=(0,(4,3))); ax.axvline(n_tr+n_va, color=MUTED, lw=0.9, ls=(0,(4,3)))
ax.set_xlabel("време (часови)", fontsize=9, color=INK2); ax.set_ylabel("број на корисници", fontsize=9, color=INK2)
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color(BASE)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
ax.set_title("Хронолошка поделба 70/10/20 (станица bs_6260)", fontsize=10.5, color=INK, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_split.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 6. transformer block
fig, ax = plt.subplots(figsize=(4.6, 6.4)); blank(ax)
blocks = [
    ("Влезна секвенца", "#ffffff", BASE),
    ("Вгнездување (embedding)\n+ позициско кодирање", "#f0f4fb", BLUE),
    ("Multi-Head\nSelf-Attention", "#f4f2fb", VIOLET),
    ("Собирање + LayerNorm", "#f7f7f5", MUTED),
    ("Feed-Forward мрежа\n(две линеарни трансформации)", "#eef7f2", AQUA),
    ("Собирање + LayerNorm", "#f7f7f5", MUTED),
    ("Излезна репрезентација", "#ffffff", BASE),
]
hh = 0.105; gap = (1 - len(blocks)*hh - 0.04) / (len(blocks)-1)
ys = []
for i, (t, fc, ec) in enumerate(blocks):
    y = 1 - 0.02 - (i+1)*hh - i*gap
    ys.append(y)
    box(ax, 0.13, y, 0.74, hh, t, fc=fc, ec=ec, fs=9.3)
    if i > 0:
        arrow(ax, 0.5, ys[i-1], 0.5, y + hh, lw=1.5)
arrow(ax, 0.13, ys[1]+hh/2, 0.05, ys[1]+hh/2, style="-")
arrow(ax, 0.05, ys[1]+hh/2, 0.05, ys[3]+hh/2, style="-")
arrow(ax, 0.05, ys[3]+hh/2, 0.13, ys[3]+hh/2)
arrow(ax, 0.87, ys[3]+hh/2, 0.95, ys[3]+hh/2, style="-")
arrow(ax, 0.95, ys[3]+hh/2, 0.95, ys[5]+hh/2, style="-")
arrow(ax, 0.95, ys[5]+hh/2, 0.87, ys[5]+hh/2)
ax.text(0.03, (ys[1]+ys[3])/2+hh/2, "резидуална\nврска", fontsize=7.5, color=MUTED, ha="center", rotation=90)
ax.text(0.975, (ys[3]+ys[5])/2+hh/2, "резидуална\nврска", fontsize=7.5, color=MUTED, ha="center", rotation=90)
ax.text(0.5, 0.005, "× N слоеви", fontsize=9, color=INK2, ha="center", style="italic")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_transformer.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 7. PatchTST
fig, ax = plt.subplots(figsize=(9.8, 4.2)); blank(ax)
tt = np.linspace(0, 1, 120)
ss = 0.10 + 0.05*np.sin(tt*4*np.pi) + 0.02*np.sin(tt*13*np.pi)
ax.plot(0.03 + tt*0.30, 0.80 + ss, color=INK2, lw=1.5)
ax.text(0.18, 0.97, "временска серија (еден канал), seq_len = 24", fontsize=9, ha="center", color=INK)
px0 = 0.03
for i in range(6):
    r = Rectangle((px0 + i*0.052, 0.55), 0.046, 0.13, fc=LIGHTBLUE, ec=BLUE, lw=1.2)
    ax.add_patch(r)
    ax.text(px0 + i*0.052 + 0.023, 0.615, f"P{i+1}", ha="center", va="center", fontsize=8, color=INK)
ax.text(0.18, 0.47, "патеки (patch_len = 6, stride = 1) → 20 преклопувачки патеки", fontsize=8.6, ha="center", color=INK2)
arrow(ax, 0.18, 0.78, 0.18, 0.70)
box(ax, 0.40, 0.55, 0.17, 0.30, "Линеарно\nвгнездување\n(d_model = 16)", fc="#f0f4fb", ec=BLUE, fs=9)
arrow(ax, 0.335, 0.62, 0.40, 0.68)
box(ax, 0.62, 0.55, 0.17, 0.30, "Transformer\nенкодер\n(3 слоја)", fc="#f4f2fb", ec=VIOLET, fs=9)
arrow(ax, 0.57, 0.70, 0.62, 0.70)
box(ax, 0.84, 0.55, 0.13, 0.30, "Flatten +\nлинеарна\nглава", fc="#eef7f2", ec=AQUA, fs=9)
arrow(ax, 0.79, 0.70, 0.84, 0.70)
ax.text(0.905, 0.47, "прогноза (1 ч.)", fontsize=9, ha="center", color=RED)
arrow(ax, 0.905, 0.55, 0.905, 0.50, color=RED)
box(ax, 0.03, 0.06, 0.44, 0.26, "Каналска независност:\nсекој од 3-те канали (packets, bytes, users)\nминува независно низ истиот енкодер", fc="#fbf7ec", ec=YELLOW, fs=8.8)
box(ax, 0.53, 0.06, 0.44, 0.26, "RevIN (инстанциска нормализација):\nсекој влезен прозорец се нормализира,\nа прогнозата се денормализира", fc="#fdf4ec", ec=ORANGE, fs=8.8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_patchtst.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 8. CATS
fig, ax = plt.subplots(figsize=(9.8, 4.4)); blank(ax)
for i in range(6):
    r = Rectangle((0.03 + i*0.052, 0.68), 0.046, 0.13, fc=LIGHTBLUE, ec=BLUE, lw=1.2)
    ax.add_patch(r)
ax.text(0.185, 0.86, "патеки од минатото (20 × 6 ч.)", fontsize=9, ha="center", color=INK)
r = Rectangle((0.10, 0.18), 0.10, 0.13, fc="#fbdddd", ec=RED, lw=1.4)
ax.add_patch(r)
ax.text(0.15, 0.245, "Q", ha="center", va="center", fontsize=10, color=RED, fontweight="bold")
ax.text(0.15, 0.10, "учлива иднинска патека-прашалник\n(learnable future query)", fontsize=8.4, ha="center", color=INK2)
box(ax, 0.40, 0.36, 0.24, 0.30, "Cross-Attention\n(без self-attention)\n3 слоја", fc="#f4f2fb", ec=VIOLET, fs=9.5)
arrow(ax, 0.34, 0.745, 0.46, 0.66); ax.text(0.375, 0.73, "K, V", fontsize=9, color=BLUE)
arrow(ax, 0.20, 0.28, 0.44, 0.36); ax.text(0.30, 0.27, "Q", fontsize=9, color=RED)
box(ax, 0.74, 0.36, 0.16, 0.30, "Линеарна\nпроекција", fc="#eef7f2", ec=AQUA, fs=9.5)
arrow(ax, 0.64, 0.51, 0.74, 0.51)
ax.text(0.90, 0.28, "прогноза\n(првиот чекор\nод патеката)", fontsize=8.6, ha="center", color=RED)
arrow(ax, 0.86, 0.36, 0.88, 0.31, color=RED)
box(ax, 0.35, 0.03, 0.62, 0.16, "QAM (Query-Adaptive Masking): за време на тренирање прашалниците се маскираат\nсо растечка веројатност (0.1 → 0.3) за подобра генерализација", fc="#fbf7ec", ec=YELLOW, fs=8.6)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_cats.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 9. DeformableTST
fig, ax = plt.subplots(figsize=(9.8, 4.4)); blank(ax)
np.random.seed(5)
yline = 0.80
ax.plot([0.04, 0.60], [yline, yline], color=BASE, lw=2)
ref = np.linspace(0.07, 0.57, 9)
ax.scatter(ref, [yline]*9, s=34, color=MUTED, zorder=3)
off = ref + np.array([0.015, -0.02, 0.03, 0.01, -0.03, 0.02, -0.01, 0.035, -0.02])
sel = [1, 3, 4, 7]
ax.scatter(off[sel], [yline]*len(sel), s=64, color=RED, zorder=4, marker="o")
for i in sel:
    a = FancyArrowPatch((ref[i], yline+0.028), (off[i], yline+0.10), arrowstyle="-|>",
                        mutation_scale=9, color=RED, lw=1.1,
                        connectionstyle="arc3,rad=0.3")
    ax.add_patch(a)
ax.text(0.32, 0.95, "референтни точки (сиво) + учени поместувања (offsets, црвено)", fontsize=9, ha="center", color=INK)
box(ax, 0.66, 0.66, 0.31, 0.26, "Deformable Attention:\nсекое прашање посетува мал број\nдинамички избрани временски точки", fc="#f4f2fb", ec=VIOLET, fs=8.8)
box(ax, 0.04, 0.30, 0.27, 0.24, "Влезен stem\n(конв. слој, без\nнамалување, ratio = 1)", fc="#f0f4fb", ec=BLUE, fs=8.8)
box(ax, 0.37, 0.30, 0.27, 0.24, "1 фаза × 3 блока\n[D, D, D]\n+ LPU (депт. конв.)", fc="#f4f2fb", ec=VIOLET, fs=8.8)
box(ax, 0.70, 0.30, 0.27, 0.24, "Flatten глава\n+ LayerNorm\n→ прогноза (1 ч.)", fc="#eef7f2", ec=AQUA, fs=8.8)
arrow(ax, 0.31, 0.42, 0.37, 0.42); arrow(ax, 0.64, 0.42, 0.70, 0.42)
box(ax, 0.04, 0.03, 0.93, 0.17, "Тренирање по оригиналниот рецепт: AdamW (weight decay = 0.05), 5 епохи линеарно загревање\n+ косинусно опаѓање на стапката на учење; RevIN нормализација на влезот", fc="#fbf7ec", ec=YELLOW, fs=8.6)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_deformable.png", dpi=170, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- 10. Peri-midFormer
fig, ax = plt.subplots(figsize=(9.8, 4.8)); blank(ax)
box(ax, 0.03, 0.72, 0.20, 0.20, "Временска серија\n+ временски\nобележја", fc="#ffffff", ec=BASE, fs=8.8)
box(ax, 0.29, 0.72, 0.20, 0.20, "Декомпозиција\nтренд + сезонска\n(moving_avg = 25)", fc="#f0f4fb", ec=BLUE, fs=8.8)
box(ax, 0.55, 0.72, 0.20, 0.20, "FFT анализа:\nтоп-2 доминантни\nпериоди", fc="#eef7f2", ec=AQUA, fs=8.8)
arrow(ax, 0.23, 0.82, 0.29, 0.82); arrow(ax, 0.49, 0.82, 0.55, 0.82)
levels = [(0.32, 0.52, 1), (0.26, 0.40, 2), (0.20, 0.28, 4)]
cx = 0.38
for i, (wid, y, nseg) in enumerate(levels):
    total = wid
    seg = total / nseg
    for j in range(nseg):
        r = Rectangle((cx - total/2 + j*seg + 0.004, y), seg - 0.008, 0.09,
                      fc=["#cde2fb", "#9ec5f4", "#6da7ec"][i], ec=BLUE, lw=1.0)
        ax.add_patch(r)
ax.text(cx, 0.635, "периодична пирамида (нивоа = периодични компоненти)", fontsize=9, ha="center", color=INK)
arrow(ax, 0.65, 0.72, 0.52, 0.60)
box(ax, 0.68, 0.28, 0.29, 0.30, "Attention низ и меѓу\nнивоата на пирамидата\n(3 слоја)\n→ Flatten + линеарна глава", fc="#f4f2fb", ec=VIOLET, fs=8.8)
arrow(ax, 0.56, 0.42, 0.68, 0.42)
box(ax, 0.03, 0.03, 0.60, 0.15, "Трендот се прогнозира со посебен линеарен слој;\nсезонскиот дел — преку пирамидата. Излез: label_len + pred_len чекори", fc="#fbf7ec", ec=YELLOW, fs=8.4)
ax.text(0.82, 0.16, "прогноза (1 ч.)", fontsize=9.5, ha="center", color=RED)
arrow(ax, 0.82, 0.28, 0.82, 0.21, color=RED)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_perimidformer.png", dpi=170, bbox_inches="tight"); plt.close()

print("diagrams done")
