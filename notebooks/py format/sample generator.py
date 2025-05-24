# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:12:03 2025

@author: cangu
"""

#This code was used to take 500 samples from a large datasets 
#and scrape the information we need and prpare a tidy dataset.

import pandas as pd
import numpy as np

# --- Load raw file ---------------------------------------------------------
raw = pd.read_csv("transfermarkt_fbref_201920.csv",
                  sep=';', encoding='utf-8', on_bad_lines='skip')

# --- Select & engineer columns --------------------------------------------
tidy = pd.DataFrame({
    "player"      : raw["player"],
    "nationality" : raw["nationality"].str[-3:],        # keep ISO‑3 code
    "age"         : raw["age"],
    "league"      : raw["league"],
    "market_value_eur": raw["value"],                   # already numeric €
    "score_contrib_per90":
         raw["goals_per90"].fillna(0) + raw["assists_per90"].fillna(0),
    "cards_2019_20":
         raw["cards_yellow"].fillna(0) + raw["cards_red"].fillna(0),
    # placeholders for data you’ll scrape later
    "avg_match_rating": np.nan,
    "injury_days_total": np.nan,
    "married": np.nan
})

# --- Keep only 500 players -------------------------------------------------
tidy = tidy.sample(500, random_state=42)   # reproducible sample
tidy.reset_index(drop=True, inplace=True)

# --- Save for EDA ----------------------------------------------------------
tidy.to_csv("players_2019_20_tidy500.csv", index=False)
print("Saved:", tidy.shape)
