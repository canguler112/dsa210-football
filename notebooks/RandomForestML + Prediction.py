from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import pandas as pd

# 1) Load your enriched data
ROOT     = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "processed" / "players_2019_20_tidy500_with_avg_match_rating_v3.csv"
IMG_DIR  = ROOT / "images"
IMG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_CSV)

# 2) Define X & y (include avg_match_rating)
num_feats = ["age", "score_contrib_per90", "cards_2019_20", "avg_match_rating"]
cat_feats = ["league", "nationality"]

X = df[num_feats + cat_feats]
y = df["market_value_eur"]

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4) Build pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
], remainder="passthrough")  # leaves num_feats (including avg_match_rating) as-is

pipeline = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(random_state=42))
])

# 5) Train & predict
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

# 6) Evaluate
print("R²:",  r2_score(y_test, pred).round(3))
print("MAE:", mean_absolute_error(y_test, pred), "€")

# 7) Example prediction
new_player = pd.DataFrame([{
    "age": 24,
    "score_contrib_per90": 0.5,
    "cards_2019_20": 5,
    "avg_match_rating": 7.2,
    "league": "Serie A",
    "nationality": "Brazil"
}])

predicted_value = pipeline.predict(new_player)
print(f"Predicted market value: €{predicted_value[0]:,.0f}")
