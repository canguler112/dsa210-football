# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:20:30 2025

@author: cangu
"""

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
df = pd.read_csv(r"C:\Users\cangu\Downloads\players_2019_20_tidy500.csv")

df.info()
df.describe(include='all').T                              # quick overview
assert df.isna().sum()['market_value_eur'] == 0           # sanity check
