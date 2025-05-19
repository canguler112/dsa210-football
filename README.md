# DSA 210 Football Player Valuation Project

## 1. Project Proposal
Analyzing the Determinants of Football Player Market Value

### Motivation
Football player market valuations can vary greatly depending on a variety of criteria, including on-field performance, age, position, league quality, and more. By focusing on these characteristics, I hope to get insight into which factors have the greatest influence on how players are valued in the transfer market. This study will not only help me improve my data science skills, but it may also provide insights for teams, scouts, and fans looking to evaluate player worth more objectively.

### Features/ Factors
1. Market Value (Prediction)
2. Age
3. Nationality
4. Average Score Contribution (Goals + Assists)
5. Player League
6. Average Card per Season
7. Average Match Rating
   
### Data Source and Collection

#### Data Sources:
Transfermarkt for estimated market values, player profiles (age, position, nationality).
FBref (or WhoScored) for player performance metrics such as goals, assists, minutes played, and passing accuracy.
Potential Kaggle datasets if they cover similar data in a ready-to-use CSV format.

#### Collection Plan:
##### Scraping: 
Use Python libraries to extract market values and stats from Transfermarkt/FBref, ensuring compliance with their terms of service or manually extracting data from them.
##### CSV Downloads: 
If I find an appropriate Kaggle dataset containing both performance statistics and market values, I may opt to download it directly.
##### Data Merging: 
Match players across sources by name and/or ID. Handle name variations (e.g., special characters, abbreviations).

### Analysis & Methods

#### Data Cleaning
- Convert financial figures (e.g., “€30m”) into numeric values (30,000,000).
- Standardize player positions (e.g., grouping “LW,” “RW,” “CF” under “Forward”).
- Address missing or inconsistent data (potentially drop players with incomplete stats or interpolate where reasonable).

#### Exploratory Data Analysis (EDA)
- Visualize distributions of market values across different positions, age brackets, and leagues.
- Calculate basic statistics (mean, median, correlations) to see if there are notable patterns (e.g., does age correlate with higher or lower values?).
- Use plots (histograms, boxplots, scatterplots) to identify outliers or interesting trends.

#### Hypothesis Testing
- Test whether certain factors (e.g., goal scoring frequency) have a statistically significant relationship with market value.
- Examine differences among positions (e.g., are forwards consistently more highly valued than defenders with comparable stats?).

#### Predictive Modeling
- Implement a regression model to predict market value based on performance metrics (goals, assists, passing accuracy, etc.), age, position, nationality.
- Conduct feature importance analysis to see which variables drive value the most.


## 2. Project structure
### Progress
* 21 Apr 2025 – Collected & cleaned 500-player sample (see `data/processed`).
* 22 Apr 2025 – Completed EDA and three hypothesis tests (see the python notebooks).

### Exploratory Analysis (18 April checkpoint)

#### Hypothesis tests

| # | Test | **H₀ (null)** | **H₁ (alternative)** | Test stat | p-value | Decision <br>(α = 0.05) |
|---|------|---------------|----------------------|-----------|---------|-------------------------|
|1|**One-way ANOVA**<br>Market value by league|All leagues share the same mean market value:<br>μ<sub>Prem</sub> = μ<sub>La Liga</sub> = μ<sub>Serie A</sub> = …|At least one league’s mean market value is different.|F = **4.76**|**< 0.001**|Reject H₀|
|2|**Spearman correlation**<br>Age vs market value|ρ = 0  (no monotonic relationship)|ρ ≠ 0  (monotonic relationship exists)|ρ = **-0.13**|**0.009**|Reject H₀|
|3|**Pearson correlation**<br>Goals + Assists / 90 vs value|r = 0  (no linear relationship)|r ≠ 0  (linear relationship exists)|r = **0.36**|**< 0.001**|Reject H₀|

### Takeaways
* **Premier League** players command the highest median value (~3× Serie A).  
* **Age** shows a modest negative association with value (ρ ≈ -0.13).  
* **Scoring contribution** per 90 is strongly, positively linked to value (r ≈ 0.36).

### Data Enrichment

- The original 500-player dataset was **manually enriched** with an `avg_match_rating` feature.
- We collected each player’s average rating for the 2019–20 season (e.g. from FBref match logs) and added it as a new numeric column.
- This enrichment provides a composite performance signal—capturing consistency, defensive work, and overall contribution—that significantly boosts model accuracy.


## 3. Machine Learning Methods & Findings (23 May checkpoint)

### Random Forest Regressor with Average Match Rating  
**Script:** `src/07_ml_with_rating.py`  
**Run command:**  

python notebooks/RandomForestML + Prediction.py
| Metric      | Value        |
| ----------- | ------------ |
| **R²**      | 0.420        |
| **MAE (€)** | 7,489,153.65 |

Example prediction:
For a 24-year-old Brazilian in Serie A with

goals+assists per 90 = 0.50

cards = 5

avg_match_rating = 7.2

Predicted market value: €38,580,325

Key finding:
Adding average match rating increased explained variance from ~3% to 42%, and reduced MAE to ~€7.5 M. Future work could include hyperparameter tuning, additional features (injury days, contract length), or ensembling with gradient boosting.

### HistGradientBoostingRegressor with Average Match Rating

**Purpose:**  
- Leverage the enriched feature set (age, score_contrib_per90, cards, avg_match_rating, league, nationality)  
- Capture non-linear interactions and complex relationships via gradient‐boosted trees  
- Compare performance against the Random Forest baseline  

**Procedure:**  
1. Standard‐scale numeric features and one‐hot encode categorical ones.  
2. Grid‐search over `max_iter=[200,400]`, `learning_rate=[0.01,0.05]`, `max_depth=[None,10,20]` using 5-fold CV.  
3. Fit the best `HistGradientBoostingRegressor` on the training set.  
4. Evaluate R² and MAE on the 30% held-out test set.  
5. Compute permutation‐based feature importances for the final model.  

**Results:**  
- **Random Forest baseline** (for comparison):  
  - R² = 0.429  
  - MAE = €7,417,406  

- **HistGradientBoostingRegressor (HGB):**  
  - Best params: `learning_rate=0.05`, `max_depth=None`, `max_iter=200`  
  - Test R² = 0.308  
  - Test MAE = €8,403,984  

- **Feature importances (permutation) saved to:**  
 ![image](https://github.com/user-attachments/assets/0c23c94f-ad3e-4046-bb58-79cb30062283)
 

**Key Finding:**  
- While the HGB model offers a robust non-linear fit, the Random Forest remains the top performer on these features (R² = 0.429 vs. 0.308).  
- Permutation importances highlight which variables (including `avg_match_rating`) drive the HGB predictions.  

### Summary of ML Findings

After enriching the dataset with **average match ratings**, we evaluated two tree-based models:

- **Random Forest Regressor** achieved **R² = 0.429** and **MAE = €7.42 M**.  
  - **Top factors:**  
    1. `avg_match_rating`  
    2. `score_contrib_per90`  
    3. `age`  

- **HistGradientBoostingRegressor** (best params: `learning_rate=0.05`, `max_iter=200`, `max_depth=None`) yielded **R² = 0.308** and **MAE = €8.40 M**.  
  - **Most important predictors** (permutation importance):  
    1. `avg_match_rating`  
    2. `score_contrib_per90`  
    3. `cards_2019_20`  

Overall, adding **avg_match_rating** provided the largest single boost in predictive power, and the Random Forest model now explains over **42% of variance** in market values with under **€7.5 M average error**.  
