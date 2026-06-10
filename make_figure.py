"""Regenerates isr_calibration_figure.png from the result JSONs in this repo.

Usage: python make_figure.py
Needs: numpy, matplotlib, and the four hotpot JSONs in the same folder (or runs/).
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt


def find(name):
    for p in (name, os.path.join("runs", name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def rows_of(name):
    r = json.load(open(find(name)))
    rows = [x for x in r["rows"] if x.get("status") != "error"]
    return rows, r["summary"]["scores"]


def auroc(y, s):
    p, n = s[y == 1], s[y == 0]
    return (p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()


def arrays(rows):
    y = np.array([1 if x["supported"] else 0 for x in rows])
    return y, {k: np.array([x[k] for x in rows]) for k in ("q_canon", "q_marg", "q_std")}


_, s05 = rows_of("isr_hotpot.json")
r15, s15 = rows_of("isr_hotpot_1p5b.json")
rk3, s_k3 = rows_of("isr_hotpot_1p5b_k3.json")
rk2, s_k2 = rows_of("isr_hotpot_1p5b_k2.json")
y, d = arrays(r15)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# Panel 1 - order effects emerge with scale
labels, x, w = ["q_canon", "q_marg", "isr"], np.arange(3), 0.35
v05 = [s05[k]["auroc"] for k in labels]
v15 = [s15[k]["auroc"] for k in labels]
ax[0].bar(x - w / 2, v05, w, label="Qwen2.5-0.5B", color="#9bb8d3")
ax[0].bar(x + w / 2, v15, w, label="Qwen2.5-1.5B", color="#2b6cb0")
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].set_ylim(0.75, 0.84)
ax[0].set_ylabel("AUROC")
ax[0].legend(loc="upper left", fontsize=9)
ax[0].set_title(
    "1. Order effects emerge with scale (hotpot)\n"
    "marg-canon: -0.003 @0.5B -> +0.025 @1.5B (P=1.000)",
    fontsize=10,
)

# Panel 2 - marginalization saturates by k=3
ks = [1, 2, 3, 6]
marg_k = [
    s15["q_canon"]["auroc"],
    s_k2["q_marg"]["auroc"],
    s_k3["q_marg"]["auroc"],
    s15["q_marg"]["auroc"],
]
y2, d2 = arrays(rk2)
y3, d3 = arrays(rk3)
std_k = [np.nan, auroc(y2, d2["q_std"]), auroc(y3, d3["q_std"]), auroc(y, d["q_std"])]
ax[1].plot(ks, marg_k, "o-", color="#2b6cb0", label="q_marg (k passes)")
ax[1].plot(ks, std_k, "D--", color="#c05621", label="q_std alone")
ax[1].axhline(marg_k[0], color="gray", lw=0.8, ls=":")
ax[1].set_xticks(ks)
ax[1].set_xlabel("verifier passes (orderings)")
ax[1].set_ylabel("AUROC")
ax[1].legend(fontsize=9)
ax[1].set_title("2. Saturates by k=3; dispersion needs k>=3\n(hotpot, 1.5B)", fontsize=10)

# Panel 3 - the dispersion penalty has the wrong sign at 1.5B
lams = np.linspace(-2, 2, 81)
aucs = [auroc(y, d["q_marg"] - l * d["q_std"]) for l in lams]
shipped = auroc(y, d["q_marg"] - d["q_std"])
ax[2].plot(lams, aucs, color="#2b6cb0")
ax[2].axvline(0, color="gray", lw=0.8, ls=":")
ax[2].scatter([1.0], [shipped], color="#c53030", zorder=5)
ax[2].annotate(
    "shipped default (lambda=+1)\n= isr 0.786",
    (1.0, shipped),
    textcoords="offset points",
    xytext=(-10, -28),
    fontsize=9,
    color="#c53030",
)
ax[2].set_xlabel("dispersion penalty lambda   (ISR = q_marg - lambda*q_std)")
ax[2].set_ylabel("AUROC")
ax[2].set_title("3. Penalty subtracts a positive signal\n(hotpot, 1.5B - optimum at lambda <= 0)", fontsize=10)

plt.tight_layout()
plt.savefig("isr_calibration_figure.png", dpi=200, bbox_inches="tight")
print("saved isr_calibration_figure.png")
