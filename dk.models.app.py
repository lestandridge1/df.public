import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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
            pts = row.get('PTS', row.get('Points', 0))
            threes = row.get('3PM', row.get('3P', 0))
            reb = row.get('REB', row.get('TRB', 0))
            ast = row.get('AST', row.get('Assists', 0))
            blk = row.get('BLK', row.get('Blocks', 0))
            stl = row.get('STL', row.get('Steals', 0))
            tov = row.get('TOV', row.get('Turnovers', 0))

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

    # --- 3. Model Selection --- #
    st.subheader("🧠 Model Selection and Role Prediction")
    model_types = ["Lasso", "Ridge", "ElasticNet"]
    st.write("Training models:", ", ".join(model_types))

    role_cols = ['Was_Captain', 'Was_UTIL1', 'Was_UTIL2', 'Was_UTIL3', 'Was_UTIL4', 'Was_UTIL5']
    available_roles = [col for col in role_cols if col in df.columns]

    if available_roles:
        features = df.select_dtypes(include=[np.number]).drop(columns=available_roles, errors='ignore').dropna(axis=1)

        X = features
        Y = df[available_roles].astype(int)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        def train_model(X_train, y_train, model_type):
            if model_type == "Lasso":
                return Lasso(alpha=0.1).fit(X_train, y_train)
            elif model_type == "Ridge":
                return Ridge(alpha=1.0).fit(X_train, y_train)
            else:
                return ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)

        predictions = {}
        st.text("Classification Reports for All Roles:")
        for role in available_roles:
            st.write(f"\n**{role}**")
            preds_all = []
            for model_type in model_types:
                model = train_model(X_train, Y_train[role], model_type)
                preds = model.predict(X_test) > 0.5
                preds_all.append(preds.astype(int))
                st.text(f"Model: {model_type}\n" + classification_report(Y_test[role], preds))

            # Ensemble Voting: majority vote
            ensemble_pred = (np.sum(preds_all, axis=0) >= 2).astype(int)
            predictions[role] = ensemble_pred
            st.text("Ensemble (Voting)\n" + classification_report(Y_test[role], ensemble_pred))

        # --- 6. Accuracy Evaluator --- #
        st.subheader("✅ Accuracy Evaluator")
        accuracy_metrics = []
        for role in available_roles:
            acc = accuracy_score(Y_test[role], predictions[role])
            accuracy_metrics.append((role, round(acc * 100, 2)))

        acc_df = pd.DataFrame(accuracy_metrics, columns=["Role", "Accuracy (%)"])
        st.table(acc_df)

        total_matches = sum([np.sum(predictions[role] == Y_test[role].values) for role in available_roles])
        total_predictions = sum([len(Y_test[role]) for role in available_roles])
        st.metric("Overall Match Accuracy", f"{round((total_matches / total_predictions) * 100, 2)}%")

    else:
        st.warning("No role columns found to train models on. Expected: Was_Captain, Was_UTIL1, ..., Was_UTIL5")
