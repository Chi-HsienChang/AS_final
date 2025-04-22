#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 1. Read A5 events and initialize columns
folder = "t_allParse_ss"
print(f"Folder path: {folder}")

a5_df = pd.read_csv("./data/A5_events_0419.csv")
a5_df["index"]    = -1
a5_df["prob_5ss"] = -1.0
a5_df["prob_3ss"] = -1.0

# 2. Compile regex patterns
sec5_re = re.compile(
    r"Sorted 5' Splice Sites \(High to Low Probability\):([\s\S]+?)Sorted 3' Splice Sites",
    re.MULTILINE
)
sec3_re = re.compile(
    r"Sorted 3' Splice Sites \(High to Low Probability\):([\s\S]+)",
    re.MULTILINE
)
pos_re = re.compile(r"Position\s*(\d+):\s*([0-9.eE+\-]+)")

# 3. Map gene to (index, file content)
gene_to_index = {}
for fname in os.listdir(folder):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(folder, fname)
    text = open(path, encoding="utf-8").read()
    mg = re.search(r"Gene\s*=\s*(\S+)", text)
    mi = re.search(r"_g_(\d+)\.txt", fname)
    if mg and mi:
        gene_to_index[mg.group(1)] = (int(mi.group(1)), text)

# 4. Extract prob_5ss and prob_3ss for each event
for i, row in a5_df.iterrows():
    gene = row["gene"]
    ss5  = int(row["5ss_alternative"])
    ss3  = int(row["3ss_paring"])
    if gene not in gene_to_index:
        continue

    idx, text = gene_to_index[gene]
    a5_df.at[i, "index"] = idx

    # 5' splice site probability
    prob5 = -1.0
    m5 = sec5_re.search(text)
    if m5:
        for line in m5.group(1).splitlines():
            m = pos_re.match(line.strip())
            if m and int(m.group(1)) == ss5:
                prob5 = float(m.group(2))
                break

    # 3' splice site probability
    prob3 = -1.0
    m3 = sec3_re.search(text)
    if m3:
        for line in m3.group(1).splitlines():
            m = pos_re.match(line.strip())
            if m and int(m.group(1)) == ss3:
                prob3 = float(m.group(2))
                break

    a5_df.at[i, "prob_5ss"] = prob5
    a5_df.at[i, "prob_3ss"] = prob3

# 5. Save and filter out events missing either score
os.makedirs("./A5_result", exist_ok=True)
a5_df.to_csv("./A5_result/A5_events_with_prob.csv", index=False)

filtered = a5_df[
    (a5_df["prob_5ss"] > 0) &
    (a5_df["prob_3ss"] > 0)
]
filtered.to_csv("./A5_result/A5_events_with_prob_only.csv", index=False)

indexes = sorted(filtered["index"].unique())
print("Matched indexes:", indexes)
print("Number of genes:", len(indexes))
print("Number of events:", len(filtered))

# 6. For each index get the max/min PSI event's prob_5ss and record PSI
records = []
for idx in indexes:
    sub = filtered[filtered["index"] == idx]
    gene = sub["gene"].iloc[0]

    max_row = sub.loc[sub["PSI"].idxmax()]
    records.append({
        "index":    idx,
        "gene":     gene,
        "PSI":      max_row["PSI"],
        "prob_5ss": max_row["prob_5ss"],
        "type":     "major"
    })

    min_row = sub.loc[sub["PSI"].idxmin()]
    records.append({
        "index":    idx,
        "gene":     gene,
        "PSI":      min_row["PSI"],
        "prob_5ss": min_row["prob_5ss"],
        "type":     "minor"
    })

df_extremes = pd.DataFrame(records)

# 7. Boxplot + swarm of major vs. minor prob_5ss
plt.figure(figsize=(6, 4))
sns.boxplot(x="type", y="prob_5ss", data=df_extremes, order=["major", "minor"],
            showfliers=False, palette={"major":"blue","minor":"orange"})
sns.swarmplot(x="type", y="prob_5ss", data=df_extremes, order=["major", "minor"],
              color=".25", size=6)
plt.xlabel("")
plt.ylabel("Alternative 5'SS Score")
plt.xticks([0, 1], ["Major 5'SS", "Minor 5'SS"])
plt.tight_layout()
plt.savefig("./A5_result/A5_maxmin_prob_swarm_box.png", dpi=300)
plt.show()

# 8. Scatter of PSI vs. prob_5ss for extremes only, with jitter to separate overlaps
jitter = 0.5
x = df_extremes["PSI"] + np.random.normal(scale=jitter, size=len(df_extremes))
y = df_extremes["prob_5ss"]
colors = df_extremes["type"].map({"major":"blue","minor":"orange"})

r, p = pearsonr(df_extremes["PSI"], df_extremes["prob_5ss"])

plt.figure(figsize=(6,6))
plt.scatter(x, y, c=colors, s=80, alpha=0.7, edgecolor="k")
plt.xlabel("PSI")
plt.ylabel("Alternative 5'SS Score")
plt.title(f"Pearson r = {r:.2f}, p = {p:.2g}")
# custom legend
plt.legend(handles=[
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', label='Major 5\'SS', markersize=8),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', label='Minor 5\'SS', markersize=8)
], title="Event")
plt.tight_layout()
plt.savefig("./A5_result/PSI_vs_prob5ss_extremes_scatter_jitter.png", dpi=300)
plt.show()
