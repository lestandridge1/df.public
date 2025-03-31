import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary
import random
import matplotlib.pyplot as plt

st.title("🏀 DraftKings NBA Lineup Optimizer")

# --- 1. Upload Interface --- #
uploaded_file = st.file_uploader("Upload your CSV or Pickle file", type=["csv", "pkl"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_pickle(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # --- 2. DraftKings Fantasy Point Calculator --- #
    stat_cols = ['PTS', '3PM', 'REB', 'AST', 'BLK', 'STL', 'TOV']
    alt_names = ['Points', '3P', 'TRB', 'Assists', 'Blocks', 'Steals', 'Turnovers']
    col_map = dict(zip(stat_cols, alt_names))
    available = [col for col in stat_cols if col in df.columns]

    if len(available) < len(stat_cols):
        renamed = False
        for orig, alt in col_map.items():
            if alt in df.columns:
                df[orig] = df[alt]
                renamed = True
        if not all(col in df.columns for col in stat_cols):
            st.warning("Missing one or more required stat columns: PTS, 3PM, REB, AST, BLK, STL, TOV")

    if all(col in df.columns for col in stat_cols):
        def compute_dk_points(row):
            try:
                pts = float(row.get('PTS', row.get('Points', 0)))
                threes = float(row.get('3PM', row.get('3P', 0)))
                reb = float(row.get('REB', row.get('TRB', 0)))
                ast = float(row.get('AST', row.get('Assists', 0)))
                blk = float(row.get('BLK', row.get('Blocks', 0)))
                stl = float(row.get('STL', row.get('Steals', 0)))
                tov = float(row.get('TOV', row.get('Turnovers', 0)))
            except Exception as e:
                return 0.0

            double_double_count = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
            triple_bonus = 3 if double_double_count >= 3 else 0
            double_bonus = 1.5 if double_double_count >= 2 else 0

            return (
                pts + 0.5 * threes + 1.25 * reb + 1.5 * ast +
                2 * blk + 2 * stl - 0.5 * tov + double_bonus + triple_bonus
            )

        df['DraftKings_FP_Calculated'] = df.apply(compute_dk_points, axis=1)
        st.subheader("📊 DraftKings Points Calculated")
        st.dataframe(df[['PTS', '3PM', 'REB', 'AST', 'BLK', 'STL', 'TOV', 'DraftKings_FP_Calculated']].head())

    # --- 3. Predict Fantasy Points with Regression Models --- #
    st.subheader("🧠 Predict Fantasy Points with Regression Models")
    model_types = ["Lasso", "Ridge", "ElasticNet"]

    if 'DraftKings_FP_Calculated' in df.columns:
        features = df.select_dtypes(include=[np.number]).drop(columns=['DraftKings_FP_Calculated'], errors='ignore')
        X = features.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df['DraftKings_FP_Calculated']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        predictions = {}
        for model_type in model_types:
            if model_type == "Lasso":
                model = Lasso(alpha=0.1)
            elif model_type == "Ridge":
                model = Ridge(alpha=1.0)
            else:
                model = ElasticNet(alpha=0.1, l1_ratio=0.5)

            model.fit(X_train, y_train)
            preds = model.predict(X)
            df[f'Predicted_FP_{model_type}'] = preds
            predictions[model_type] = preds

        df['Predicted_FP_Ensemble'] = np.mean([predictions[m] for m in model_types], axis=0)

        st.success("Fantasy point predictions completed. Added columns: Predicted_FP_Lasso, Predicted_FP_Ridge, Predicted_FP_ElasticNet, and Predicted_FP_Ensemble")
    else:
        st.warning("Missing column: DraftKings_FP_Calculated")
