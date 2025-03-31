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
        y = pd.to_numeric(df['DraftKings_FP_Calculated'], errors='coerce').fillna(0)

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

    # --- 4. Lineup Optimizer Using Predicted Fantasy Points by Series ID --- #
st.subheader("💸 Optimized Lineups Using Predicted Fantasy Points by Series ID")
if 'Series ID' in df.columns:
    unique_series = df['Series ID'].unique().tolist()
    selected_series = st.selectbox("Select a Series ID to run lineup optimization for:", unique_series)


    if 'Predicted_FP_Ensemble' in df.columns and 'Draftkings Captain Salary' in df.columns:
        top_n_lineups = st.slider("How many top lineups to generate?", min_value=1, max_value=500, value=200)

                series_df = df[df['Series ID'] == selected_series]
        lineups = []
        for _ in range(top_n_lineups * 5):
            sample = series_df.sample(n=6)
            for i in range(len(sample)):
                cap = sample.iloc[i]
                utils = sample.drop(index=cap.name)
                if len(utils) != 5:
                    continue
                cap_salary = cap['Draftkings Captain Salary']
                util_salary = (2/3) * utils['Draftkings Captain Salary'].sum()
                total_salary = cap_salary + util_salary
                if total_salary <= 50000:
                    cap_fp = 1.5 * cap['Predicted_FP_Ensemble']
                    util_fp = utils['Predicted_FP_Ensemble'].sum()
                    total_fp = cap_fp + util_fp
                    lineup = {
                        'Captain': cap.name,
                        'UTILs': utils.index.tolist(),
                        'Total_FP': total_fp,
                        'Total_Salary': total_salary
                    }
                    lineups.append(lineup)

        top_lineups = sorted(lineups, key=lambda x: -x['Total_FP'])[:top_n_lineups]
        st.write(f"Generated top {len(top_lineups)} lineups below:")

        for i, l in enumerate(top_lineups[:10]):
            st.markdown(f"**Lineup #{i+1}**")
            cap = series_df.loc[l['Captain']]
            utils = series_df.loc[l['UTILs']]
            st.write("🧢 Captain:", cap['Team'], cap['Opponent'], round(cap['Predicted_FP_Ensemble'], 2))
            st.write("🔧 UTILs:")
            st.dataframe(utils[['Team', 'Opponent', 'Predicted_FP_Ensemble']])
            st.markdown(f"**Total Predicted FP**: {round(l['Total_FP'], 2)} | **Total Salary**: {int(l['Total_Salary'])}")

        # --- 5. Optional: CSV Export & Score Comparison --- #
        # Export Top Lineups
        export_button = st.button("⬇️ Export Top Lineups to CSV")
        if export_button:
            export_data = []
            for idx, l in enumerate(top_lineups):
                cap_row = series_df.loc[[l['Captain']]].copy()
                cap_row['Role'] = 'Captain'
                util_rows = series_df.loc[l['UTILs']].copy()
                util_rows['Role'] = 'UTIL'
                lineup_df = pd.concat([cap_row, util_rows])
                if 'Starters' in lineup_df.columns:
                    lineup_df.rename(columns={'Starters': 'PlayerName'}, inplace=True)
                lineup_df['LineupRank'] = idx + 1
                lineup_df['LineupTotalPredictedFP'] = l['Total_FP']
                lineup_df['LineupTotalSalary'] = l['Total_Salary']
                export_data.append(lineup_df)

            export_csv = pd.concat(export_data).to_csv(index=False).encode('utf-8')
            st.download_button("Download Top Lineups CSV", data=export_csv, file_name="top_lineups.csv", mime="text/csv")

        # Show Predicted vs Actual Fantasy Points
        st.subheader("📈 Predicted vs Actual Fantasy Points (Top 10 Lineups)")
        comparison_rows = []
        for l in top_lineups[:10]:
            cap = series_df.loc[l['Captain']]
            utils = series_df.loc[l['UTILs']]
            predicted = 1.5 * cap['Predicted_FP_Ensemble'] + utils['Predicted_FP_Ensemble'].sum()
            actual = 1.5 * cap['DraftKings_FP_Calculated'] + utils['DraftKings_FP_Calculated'].sum()
            comparison_rows.append({
                'Captain': cap['Team'],
                'Total_Predicted_FP': predicted,
                'Total_Actual_FP': actual,
                'Difference': actual - predicted
            })
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df)

        # Optional: Check for ground truth match if Was_Captain and Was_UTIL* are available
        role_cols = ['Was_Captain', 'Was_UTIL1', 'Was_UTIL2', 'Was_UTIL3', 'Was_UTIL4', 'Was_UTIL5']
        series_df.columns = series_df.columns.str.strip().str.replace('?', '', regex=False)
        if all(col in df.columns for col in role_cols):
            true_lineup_ids = series_df[series_df['Was_Captain'] == 1].index.tolist() + series_df[[f'Was_UTIL{i}' for i in range(1,6)]].stack().reset_index().query('0 == 1')['level_0'].tolist()
            true_lineup_ids = list(set(true_lineup_ids))
            found_rank = None
            for idx, l in enumerate(top_lineups):
                lineup_ids = [l['Captain']] + l['UTILs']
                if set(lineup_ids) == set(true_lineup_ids):
                    found_rank = idx + 1
                    break
            if found_rank:
                st.success(f"✅ Found historical true lineup in top {found_rank} predicted lineups!")
            else:
                st.info("❌ Historical true lineup was NOT found in the top predicted lineups.")
