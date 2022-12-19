import pandas as pd
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")

n_simulation = 100
min_operating_time = 30

simulation_date = date.today().strftime("%b-%d-%Y")

### Data preparation
df_l = []

for generator in range(1, 5):

    df = pd.read_csv(f"data/prepared/ap{generator}_data.csv")
    df["time"] = df.groupby("Run").cumcount(ascending=True)
    df["generator"] = generator
    df.rename(columns={"Vazão Turbinado": "Vazão Turbinada"}, inplace=True)
    df_l.append(df)

df = pd.concat(df_l)

df["id"] = df["generator"].astype(str) + "-" + df["Run"].astype(str)

data = (
    df.dropna()
    .groupby("id")
    .agg(
        {
            "Run": "median",
            "Breakdown": "median",
            "time": "max",
        }
    )
)

data = data[data["time"] > min_operating_time]  # Select only runs with time
data["Breakdown"] = data["Breakdown"].astype(bool)


# Simulation
simulation_l = []

for features in ["hos_features", "tsfresh_features"]:

    # Load features
    if features == "hos_features":
        df_feature = pd.read_csv(
            "data/preprocessed/hos_features_rev2.zip",
            compression="zip",
            index_col=0,
            skiprows=[1, 2],
            # usecols=range(209),
        )

    else:
        df_feature = pd.read_csv(
            "data/preprocessed/tsfresh_features_sel.zip", index_col=0, compression="zip"
        )

    df_feature = df_feature.reindex(data.index)

    y = np.zeros(len(data), dtype={"names": ("cens", "time"), "formats": ("?", "<f8")})
    y["cens"] = data["Breakdown"].tolist()
    y["time"] = data["time"].tolist()

    X = df_feature.values

    for random_state in tqdm(range(n_simulation)):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=3 * random_state
        )

        models = [
            ("CoxNet", CoxnetSurvivalAnalysis()),
            (
                "RSF",
                RandomSurvivalForest(
                    n_estimators=100,
                    min_samples_split=10,
                    min_samples_leaf=15,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=7 * random_state,
                ),
            ),
            (
                "GBSA",
                GradientBoostingSurvivalAnalysis(
                    n_estimators=100,
                    min_samples_split=10,
                    min_samples_leaf=15,
                    learning_rate=1.0,
                    random_state=11 * random_state,
                ),
            ),
        ]

        for model_name, model in models:

            start = time.time()

            model.fit(X_train, y_train)

            fit_time = time.time() - start

            score = model.score(X_test, y_test).round(5)

            simulation_l.append(
                {
                    "features": features,
                    "run": random_state,
                    "model": model_name,
                    "score": score,
                    "time": fit_time,
                }
            )

df_results = pd.DataFrame(simulation_l)
df_results.to_csv(f"data/results/{simulation_date}.csv", index=False)
