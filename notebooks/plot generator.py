# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:29:26 2025

@author: cangu
"""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

