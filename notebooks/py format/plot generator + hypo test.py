# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:29:26 2025

@author: cangu
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, spearmanr, pearsonr

sns.set_theme(style="darkgrid")           
df = pd.read_csv(r"C:\Users\cangu\Downloads\players_2019_20_tidy500.csv")

display(df.describe().T) 

plt.figure(figsize=(6,4))
sns.histplot(df['market_value_eur'], bins=40, log_scale=True)
plt.title('Distribution of player market values (€) – log scale')
plt.xlabel('Market value (€)'); plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x='league', y='market_value_eur', data=df)
plt.yscale('log')
plt.title('Market value by league')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='market_value_eur', data=df, alpha=.5)
sns.regplot(x='age', y='market_value_eur', data=df,
            scatter=False, color='black')   
plt.yscale('log'); plt.title('Age vs market value')
plt.show()

sns.lmplot(x='score_contrib_per90', y='market_value_eur',
           data=df, scatter_kws={'alpha':0.5})
plt.yscale('log'); plt.title('Goals+Assists per 90 vs market value')
plt.show()

# ---------------- H₁: Market value differs by league ----------------
groups = [g['market_value_eur'].values for _, g in df.groupby('league')]
F, p_aov = f_oneway(*groups)
print(f"ANOVA across leagues  →  F = {F:.2f},  p = {p_aov:.4g}")

# ---------------- H₂: Age vs market value (rank‑based) --------------
rho_age, p_age = spearmanr(df['age'], df['market_value_eur'])
print(f"Spearman ρ (age,value) →  ρ = {rho_age:.2f},  p = {p_age:.4g}")

# ---------------- H₃: Score contrib vs market value -----------------
r_sc, p_sc = pearsonr(df['score_contrib_per90'], df['market_value_eur'])
print(f"Pearson r (GA/90,value) →  r = {r_sc:.2f},  p = {p_sc:.4g}")

alpha = 0.05      # significance level

print("ANOVA:",      "reject H0" if p_aov < alpha else "fail to reject H0")
print("Age value:",  "reject H0" if p_age < alpha else "fail to reject H0")
print("GA/90 value:", "reject H0" if p_sc  < alpha else "fail to reject H0")

""" Statistical findings
1st test: 
   H₀ (null): All leagues share the same mean market value
   H₁ (alternative): At least one league’s mean market value is different.
   H(1) (ANOVA):** p < 0.05 ⇒ at least one league’s market values differ significantly. Premier League median ≈3× Serie A.

2nd test:
   H₀: ρ = 0 (No monotonic relationship between age and market value.)
   H₁: ρ ≠ 0 (A monotonic relationship exists.)
   H(2) (Spearman ρ = ‑0.13, p < 0.05):** older age correlates with lower market value.

3rd test:
    H₀: r = 0 (No linear relationship between scoring contribution and value.)
    H₁: r ≠ 0 (A linear relationship exists.)
    H(3) (Pearson r = 0.36, p < 0.05):** higher goals+assists / 90 strongly associated with higher value."""
