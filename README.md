# DSA 210 Football Player Valuation Project

## 1. Project Proposal
Analyzing the Determinants of Football Player Market Value

### Motivation
Football player market valuations can vary greatly depending on a variety of criteria, including on-field performance, age, position, league quality, and more. By focusing on these characteristics, I hope to get insight into which factors have the greatest influence on how players are valued in the transfer market. This study will not only help me improve my data science skills, but it may also provide insights for teams, scouts, and fans looking to evaluate player worth more objectively.

### Features/ Factors
1. Market Value
2. Age
3. Nationality
4. Average Score Contribution (Goals + Assists)
5. Average Match Rating (1-10)
6. Player League
7. Average Card per Season
8. Total Time Spent Injured
9. Marriage Status

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

> _Limitations_: Dataset covers only 2019-20; match-rating, injury-days, and marital-status columns are placeholders for future scraping.
