import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary
import random

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
    if all(col in df.columns for col in stat_cols):

        def compute_dk_points(row):
            stats = [row['PTS'], row['3PM'], row['REB'], row['AST'], row['BLK'], row['STL'], row['TOV']]
            double_double_count = sum([row['PTS'] >= 10, row['REB'] >= 10, row['AST'] >= 10, row['STL'] >= 10, row['BLK'] >= 10])
            triple_bonus = 3 if double_double_count >= 3 else 0
            double_bonus = 1.5 if double_double_count >= 2 else 0
            return (
                row['PTS'] + 0.5 * row['3PM'] + 1.25 * row['REB'] + 1.5 * row['AST'] +
                2 * row['BLK'] + 2 * row['STL'] - 0.5 * row['TOV'] + double_bonus + triple_bonus
            )

        df['DraftKings_FP_Calculated'] = df.apply(compute_dk_points, axis=1)
        st.subheader("📊 DraftKings Points Calculated")
        st.dataframe(df[['PTS', '3PM', 'REB', 'AST', 'BLK', 'STL', 'TOV', 'DraftKings_FP_Calculated']].head())
    else:
        st.warning("Missing one or more required stat columns: PTS, 3PM, REB, AST, BLK, STL, TOV")

    # --- 3. Model Selection --- #
    st.subheader("🧠 Model Selection and Role Prediction")
    model_type = st.selectbox("Select Model", ["Lasso", "Ridge", "ElasticNet"])

    role_cols = ['Was_Captain', 'Was_UTIL1', 'Was_UTIL2', 'Was_UTIL3', 'Was_UTIL4', 'Was_UTIL5']
    available_roles = [col for col in role_cols if col in df.columns]

    if available_roles:
        features = df.select_dtypes(include=[np.number]).drop(columns=available_roles, errors='ignore').dropna(axis=1)

        # Multi-output prediction
        X = features
        Y = df[available_roles].astype(int)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        def train_model(X_train, y_train):
            if model_type == "Lasso":
                return Lasso(alpha=0.1).fit(X_train, y_train)
            elif model_type == "Ridge":
                return Ridge(alpha=1.0).fit(X_train, y_train)
            else:
                return ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)

        predictions = {}
        st.text("Classification Reports for All Roles:")
        for role in available_roles:
            model = train_model(X_train, Y_train[role])
            preds = model.predict(X_test) > 0.5
            predictions[role] = preds
            st.write(f"\n**{role}**")
            st.text(classification_report(Y_test[role], preds))

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

    # --- 7. Top-N Lineup Match Evaluator --- #
    st.subheader("📈 Top-N Lineup Match Evaluator")
    if 'Series ID' in df.columns and 'DraftKings_FP_Calculated' in df.columns and 'Draftkings Captain Salary' in df.columns:
        top_n = st.slider("Select Top-N range to evaluate", min_value=1, max_value=50, value=10)
        df['Season'] = df['Series ID'].astype(str).str[:4]
        team_filter = st.selectbox("Filter by Team (optional)", options=["All"] + sorted(df['Team'].dropna().unique().tolist()))
        season_filter = st.selectbox("Filter by Season (optional)", options=["All"] + sorted(df['Season'].dropna().unique().tolist()))

        match_ranks = []

                filtered_df = df.copy()
        if team_filter != "All":
            filtered_df = filtered_df[filtered_df['Team'] == team_filter]
        if season_filter != "All":
            filtered_df = filtered_df[filtered_df['Season'] == season_filter]

        grouped = filtered_df.groupby("Series ID")
        for series_id, group in grouped:
            players = group.copy()
            players = players.dropna(subset=['DraftKings_FP_Calculated', 'Draftkings Captain Salary'])

            if players.shape[0] < 6:
                continue  # skip if not enough players

            candidates = []
            for cap_idx in players.index:
                cap_row = players.loc[cap_idx]
                util_pool = players.drop(index=cap_idx)
                util_combos = util_pool.sample(min(5, len(util_pool)))  # quick sample, full combo gen is heavy

                util_salary = (2/3) * util_combos['Draftkings Captain Salary'].sum()
                total_salary = cap_row['Draftkings Captain Salary'] + util_salary
                if total_salary > 50000:
                    continue

                total_fp = 1.5 * cap_row['DraftKings_FP_Calculated'] + util_combos['DraftKings_FP_Calculated'].sum()
                lineup_ids = [cap_idx] + util_combos.index.tolist()
                candidates.append((total_fp, lineup_ids))

            candidates = sorted(candidates, key=lambda x: -x[0])

            # Find ground truth lineup
            ground_truth = players[players[role_cols].sum(axis=1) > 0].index.tolist()
            if len(ground_truth) == 6:
                for rank, (_, lineup_ids) in enumerate(candidates[:top_n]):
                    if set(lineup_ids) == set(ground_truth):
                        match_ranks.append(rank + 1)
                        break
                else:
                    match_ranks.append(None)  # not found

        st.write("Found Lineups at Ranks (if matched within Top-N):")
        st.write(match_ranks)
        found_count = sum(1 for x in match_ranks if x is not None)
        total = len(match_ranks)
        st.metric("Match Rate in Top-N", f"{found_count}/{total} ({round((found_count/total)*100, 2)}%)")
    else:
        st.warning("Missing required columns: Series ID, DraftKings_FP_Calculated, or Draftkings Captain Salary")

        # --- 8. Match Rank Distribution Visualization & Export --- #
    if match_ranks:
        import matplotlib.pyplot as plt

        st.subheader("📊 Match Rank Distribution")
        valid_ranks = [r for r in match_ranks if r is not None]
        if valid_ranks:
            fig, ax = plt.subplots()
            ax.hist(valid_ranks, bins=range(1, max(valid_ranks)+2), align='left', rwidth=0.8)
            ax.set_xlabel("Rank Where Match Occurred")
            ax.set_ylabel("Number of Contests")
            ax.set_title("Distribution of Correct Lineup Rank Position")
            st.pyplot(fig)
                else:
            st.info("No matches found in Top-N for visualization.")

        # Export Top-N Lineups CSV
        export_n = st.number_input("How many top lineups to export?", min_value=1, max_value=100, value=10)
        export_lineups = candidates[:export_n]
        export_rows = []
        for i, (fp, lineup_ids) in enumerate(export_lineups):
            lineup_data = df.loc[lineup_ids].copy()
            lineup_data['Lineup_Rank'] = i + 1
            lineup_data['Projected_Total_FP'] = fp
            export_rows.append(lineup_data)

        if export_rows:
            export_df = pd.concat(export_rows)
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Top Lineups CSV", data=csv, file_name="top_draftkings_lineups.csv", mime='text/csv')

    # (Remaining code unchanged: Salary Optimizer, Monte Carlo Sim)
    # [code continues...]
