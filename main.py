import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (GroupKFold, GridSearchCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_predict

from classes import TimeConstrainedPolynomial

df_initial=pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))

#===============
#Inspect Initial Conditions
#===============

print(df_initial.head())
print(df_initial["Glucose [mM]"].unique())
print(df_initial["Glutamine [mM]"].unique())
print(df_initial["VCD [cells/mL]"].unique())
print(df_initial.shape)

#==============
#Define Training Conditions
#==============

glucose_levels=[5,20,35]
glutamine_levels=[1,4,7]
vcd_levels=[200000, 800000, 1400000]

df_selected_conditions = df_initial[
    (df_initial["Glucose [mM]"].isin(glucose_levels))
    &
    (df_initial["Glutamine [mM]"].isin(glutamine_levels))
    &
    (df_initial["VCD [cells/mL]"].isin(vcd_levels))
]

print(df_selected_conditions.shape)


df_unique_conditions = (
    df_selected_conditions
    .groupby(
        [
            "Glucose [mM]",
            "Glutamine [mM]",
            "VCD [cells/mL]"
        ],
        as_index=False
    )
    .first()
)

df_center = df_initial[
    (df_initial["Glucose [mM]"] == 20)
    &
    (df_initial["Glutamine [mM]"] == 4)
    &
    (df_initial["VCD [cells/mL]"] == 800000)
]

selected_batches=(
    df_unique_conditions["Batch ID"].tolist()
)

center_batches = df_center["Batch ID"].tolist()

selected_batches.extend(
    batch_id for batch_id in center_batches
    if batch_id not in selected_batches
)

df_corner = df_initial[
    (df_initial["Glucose [mM]"] == 35)
    & (df_initial["Glutamine [mM]"] == 7)
    & (df_initial["VCD [cells/mL]"] == 1400000)
]

additional_batch = next(
    batch_id
    for batch_id in df_corner["Batch ID"]
    if batch_id not in selected_batches
)

selected_batches.append(additional_batch)

assert len(selected_batches) == 30
assert len(set(selected_batches)) == 30

# =============================================================================
# LOAD SELECTED BATCH FILES
# =============================================================================

batch_data = []

for batch_id in selected_batches:

    filepath = os.path.join(
        "datasets",
        f"batch_{int(batch_id):04d}.csv"
    )

    df_batch = pd.read_csv(filepath)

    df_batch["Batch ID"] = batch_id

    batch_data.append(df_batch)

df_combined = pd.concat(
    batch_data,
    ignore_index=True
)

print(df_combined.head())
print(df_combined.shape)
print(df_combined["Batch ID"].nunique())

# =============================================================================
# MERGE INITIAL CONDITIONS
# =============================================================================

df_initial_features = df_initial.rename(
    columns={
        "Glucose [mM]": "Initial Glucose [mM]",
        "Glutamine [mM]": "Initial Glutamine [mM]",
        "VCD [cells/mL]": "Initial VCD [cells/mL]"
    }
)

df_combined = df_combined.merge(
    df_initial_features,
    on="Batch ID",
    how="left",
    validate="many_to_one"
)

print(df_combined.head())

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

df_combined["Time [d]"] = (
    df_combined["Time [h]"] / 24
)

df_combined["Cell Culture Cost [$/L]"] = (
    10 * df_combined["Initial Glucose [mM]"]
    + 20 * df_combined["Initial Glutamine [mM]"]
    + 0.0005 * df_combined["Initial VCD [cells/mL]"]
)

df_combined["Cell Culture Value [$/L]"] = (
    1000 * df_combined["mAb [mg/L]"]
)

df_combined["Profit [$/L]"] = (
    df_combined["Cell Culture Value [$/L]"]
    - df_combined["Cell Culture Cost [$/L]"]
)

df_combined["Space-Time Profit [$/L/d]"] = (
    df_combined["Profit [$/L]"]
    / (df_combined["Time [d]"] + 1)
)

df_combined["Space-Time Yield [mg/L/d]"] = (
    df_combined["mAb [mg/L]"]
    / (df_combined["Time [d]"] + 1)
)

# =============================================================================
# DEFINE FEATURES AND TARGET
# =============================================================================

feature_columns = [
    "Initial Glucose [mM]",
    "Initial Glutamine [mM]",
    "Initial VCD [cells/mL]",
    "Time [h]"
]

X = df_combined[feature_columns]

y = df_combined["mAb [mg/L]"]

groups = df_combined.groupby(
    [
        "Initial Glucose [mM]",
        "Initial Glutamine [mM]",
        "Initial VCD [cells/mL]"
    ]
).ngroup()


# =============================================================================
# MODEL TRAINING AND COMPARISON
# =============================================================================

cv = GroupKFold(n_splits=5)

models = {
    "Polynomial Regression": (
        make_pipeline(
            StandardScaler(),
            PolynomialFeatures(include_bias=False),
            LinearRegression()
        ),
        {
            "polynomialfeatures__degree": [1, 2, 3, 4]
        }
    ),

    "KNN Regression": (
        make_pipeline(
            StandardScaler(),
            KNeighborsRegressor()
        ),
        {
            "kneighborsregressor__n_neighbors": [3, 5, 10, 20],
            "kneighborsregressor__weights": ["uniform", "distance"]
        }
    )
}

comparison_results = []
best_models = {}
results = []

for name, (pipeline, parameters) in models.items():

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=cv,
        scoring={
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        refit="RMSE",
        n_jobs=-1
    )

    grid.fit(X, y, groups=groups)

    best_models[name] = grid.best_estimator_

    results.append({
        "Model": name,
        "CV RMSE [mg/L]": -grid.best_score_,
        "Best Parameters": grid.best_params_
    })

    for i, params in enumerate(grid.cv_results_["params"]):

        comparison_results.append({
            "Model": name,
            "Degree": params.get("polynomialfeatures__degree"),
            "Neighbors (k)": params.get(
                "kneighborsregressor__n_neighbors"
            ),
            "Weights": params.get(
                "kneighborsregressor__weights"
            ),
            "CV RMSE [mg/L]": round(
                -grid.cv_results_["mean_test_RMSE"][i], 4
            ),
            "CV R²": round(
                grid.cv_results_["mean_test_R2"][i], 4
            )
        })

# Create model comparison table
df_comparison = pd.DataFrame(comparison_results)

df_comparison = df_comparison.sort_values(
    "CV RMSE [mg/L]"
).reset_index(drop=True)

df_comparison = df_comparison.fillna("—")

print("\nMODEL COMPARISON")
print(df_comparison.to_string(index=False))

df_comparison.to_csv(
    os.path.join("tables", "model_comparison.csv"),
    index=False
)

# Select best model
df_results = pd.DataFrame(results)

best_model_name = df_results.loc[
    df_results["CV RMSE [mg/L]"].idxmin(),
    "Model"
]

best_model = best_models[best_model_name]

print("\nBest Model:", best_model_name)

# =============================================================================
# COMPARE PHYSICALLY CONSTRAINED POLYNOMIAL MODELS
# =============================================================================

constrained_models = {
    f"Constrained Degree {degree}": make_pipeline(
        TimeConstrainedPolynomial(
            degree=degree,
            time_column="Time [h]"
        ),
        StandardScaler(with_mean=False),
        Ridge(alpha=1.0, fit_intercept=False)
    )
    for degree in [2, 3, 4]
}

comparison_results = []

all_models = {
    "Original Degree 4": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=4, include_bias=False),
        LinearRegression()
    ),
    **constrained_models
}

for name, model in all_models.items():

    predictions = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=GroupKFold(n_splits=5),
        n_jobs=-1
    )

    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    early_mask = df_combined["Time [h]"] <= 24

    early_mae = np.mean(
        np.abs(
            y[early_mask].to_numpy()
            - predictions[early_mask]
        )
    )

    comparison_results.append({
        "Model": name,
        "CV RMSE [mg/L]": rmse,
        "CV R²": r2,
        "Early MAE [mg/L]": early_mae
    })

df_comparison = pd.DataFrame(comparison_results)

df_comparison = df_comparison.sort_values(
    "CV RMSE [mg/L]"
)

print("\nPHYSICALLY CONSTRAINED MODEL COMPARISON")
print(
    df_comparison.round(4).to_string(index=False)
)

# =============================================================================
# SELECT BEST MODEL
# =============================================================================

best_model_name = df_results.loc[
    df_results["CV RMSE [mg/L]"].idxmin(),
    "Model"
]

best_model = best_models[best_model_name]

print("Best Model:", best_model_name)

# =============================================================================
# COMPARE EARLY-TIME PREDICTIONS
# =============================================================================

best_constrained_name = (
    df_comparison[
        df_comparison["Model"].str.startswith("Constrained")
    ]
    .iloc[0]["Model"]
)

models_to_plot = {
    "Original Degree 4": all_models["Original Degree 4"],
    best_constrained_name: all_models[best_constrained_name]
}

plt.figure(figsize=(9, 5))

actual_early = (
    df_combined[df_combined["Time [h]"] <= 72]
    .groupby("Time [h]")["mAb [mg/L]"]
    .mean()
)

plt.plot(
    actual_early.index,
    actual_early.values,
    "ko-",
    label="Actual Mean mAb"
)

for name, model in models_to_plot.items():

    predictions = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=GroupKFold(n_splits=5),
        n_jobs=-1
    )

    df_plot = df_combined.copy()
    df_plot["Prediction"] = predictions

    mean_predictions = (
        df_plot[df_plot["Time [h]"] <= 72]
        .groupby("Time [h]")["Prediction"]
        .mean()
    )

    plt.plot(
        mean_predictions.index,
        mean_predictions.values,
        marker="o",
        label=name
    )

plt.xlabel("Time [h]")
plt.ylabel("mAb Concentration [mg/L]")
plt.title("Early-Time Model Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join("figures","early_time_model_comparison.png"),dpi=300)

plt.close()

# =============================================================================
# GENERATE OPTIMIZATION CONDITIONS
# =============================================================================

glucose_range = sorted(df_initial["Glucose [mM]"].unique())
glutamine_range = sorted(df_initial["Glutamine [mM]"].unique())
vcd_range = sorted(df_initial["VCD [cells/mL]"].unique())

# Use observed time points within the common batch duration

# Use observed time points within the common batch duration
max_time = (
    df_combined.groupby("Batch ID")["Time [h]"]
    .max()
    .min()
)

time_range = sorted(
    df_combined.loc[
        (df_combined["Time [h]"] > 0)
        & (df_combined["Time [h]"] <= max_time),
        "Time [h]"
    ].unique()
)

# Generate all combinations
G, Q, V, T = np.meshgrid(
    glucose_range,
    glutamine_range,
    vcd_range,
    time_range,
    indexing="ij"
)

df_optimization = pd.DataFrame({
    "Initial Glucose [mM]": G.ravel(),
    "Initial Glutamine [mM]": Q.ravel(),
    "Initial VCD [cells/mL]": V.ravel(),
    "Time [h]": T.ravel()
})

# Convert hours to days for economic calculations
df_optimization["Time [d]"] = (
    df_optimization["Time [h]"] / 24
)


print("Number of candidate conditions:", len(df_optimization))

# =============================================================================
# PREDICT mAb CONCENTRATIONS
# =============================================================================


X_optimization = df_optimization[feature_columns]

# Preserve raw predictions for model diagnostics
df_optimization["Raw Predicted mAb [mg/L]"] = (
    best_model.predict(X_optimization)
)

# Prevent negative concentrations in economic calculations
df_optimization["Predicted mAb [mg/L]"] = (
    df_optimization["Raw Predicted mAb [mg/L]"].clip(lower=0)
)

negative_count = (
    df_optimization["Raw Predicted mAb [mg/L]"] < 0
).sum()

print(
    f"\nNegative predictions corrected: {negative_count}"
)


print(df_optimization.head())

# =============================================================================
# CALCULATE PREDICTED ECONOMIC PERFORMANCE
# =============================================================================

df_optimization["Cell Culture Cost [$/L]"] = (
    10 * df_optimization["Initial Glucose [mM]"]
    + 20 * df_optimization["Initial Glutamine [mM]"]
    + 0.0005 * df_optimization["Initial VCD [cells/mL]"]
)

df_optimization["Predicted Cell Culture Value [$/L]"] = (
    1000 * df_optimization["Predicted mAb [mg/L]"]
)

df_optimization["Predicted Profit [$/L]"] = (
    df_optimization["Predicted Cell Culture Value [$/L]"]
    - df_optimization["Cell Culture Cost [$/L]"]
)

df_optimization["Predicted Space-Time Profit [$/L/d]"] = (
    df_optimization["Predicted Profit [$/L]"]
    / (df_optimization["Time [d]"] + 1)
)

df_optimization["Predicted Space-Time Yield [mg/L/d]"] = (
    df_optimization["Predicted mAb [mg/L]"]
    / (df_optimization["Time [d]"] + 1)
)

# =============================================================================
# IDENTIFY OPTIMAL OPERATING CONDITIONS
# =============================================================================

optimal_index = df_optimization[
    "Predicted Space-Time Profit [$/L/d]"
].idxmax()

optimal_conditions = df_optimization.loc[optimal_index]

print("\nOPTIMAL OPERATING CONDITIONS")
print("=" * 45)

print(
    f"Initial Glucose: "
    f"{optimal_conditions['Initial Glucose [mM]']:.1f} mM"
)

print(
    f"Initial Glutamine: "
    f"{optimal_conditions['Initial Glutamine [mM]']:.1f} mM"
)

print(
    f"Initial VCD: "
    f"{optimal_conditions['Initial VCD [cells/mL]']:,.0f} cells/mL"
)

print(
    f"Harvest Time: "
    f"{optimal_conditions['Time [d]']:.2f} d"
)

print(
    f"Cell Culture Cost: "
    f"${optimal_conditions['Cell Culture Cost [$/L]']:,.2f}/L"
)

print(
    f"Predicted Cell Culture Value: "
    f"${optimal_conditions['Predicted Cell Culture Value [$/L]']:,.2f}/L"
)

print(
    f"Predicted Space-Time Profit: "
    f"${optimal_conditions['Predicted Space-Time Profit [$/L/d]']:,.2f}/L/d"
)

print(
    f"Predicted Space-Time Yield: "
    f"{optimal_conditions['Predicted Space-Time Yield [mg/L/d]']:.4f} mg/L/d"
)

# =============================================================================
# EXPORT OPTIMIZATION RESULTS
# =============================================================================

df_ranked = df_optimization.sort_values(
    "Predicted Space-Time Profit [$/L/d]",
    ascending=False
)

df_top10 = df_ranked.head(10)

print("\nTOP 10 OPERATING CONDITIONS")
print(df_top10.round(3).to_string(index=False))

# Round numerical columns for cleaner output and CSV exports
rounding = {
    "Initial Glucose [mM]": 1,
    "Initial Glutamine [mM]": 1,
    "Initial VCD [cells/mL]": 0,
    "Time [h]": 0,
    "Time [d]": 2,
    "Raw Predicted mAb [mg/L]": 4,
    "Predicted mAb [mg/L]": 4,
    "Cell Culture Cost [$/L]": 2,
    "Predicted Cell Culture Value [$/L]": 2,
    "Predicted Profit [$/L]": 2,
    "Predicted Space-Time Profit [$/L/d]": 2,
    "Predicted Space-Time Yield [mg/L/d]": 4,
}

df_optimization_rounded = df_optimization.round(rounding)
df_top10_rounded = df_top10.round(rounding)

print(df_top10_rounded.to_string(index=False))

df_optimization_rounded.to_csv(os.path.join("tables","optimization_results.csv"), index=False)
df_top10_rounded.to_csv(os.path.join("tables","top_10_conditions.csv"), index=False)

# =============================================================================
# CROSS-VALIDATED MODEL DIAGNOSTICS
# =============================================================================

y_pred = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv,
    groups=groups,
    n_jobs=-1
)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nMODEL VALIDATION")
print(f"Out-of-fold RMSE: {rmse:.4f} mg/L")
print(f"Out-of-fold R²: {r2:.4f}")

# =============================================================================
# OPTIMIZATION ROBUSTNESS ANALYSIS
# =============================================================================

rng = np.random.default_rng(42)

n_simulations = 2000

# Use out-of-fold prediction errors
df_residuals = df_combined[
    ["Batch ID", "Time [h]"]
].copy()

df_residuals["Residual [mg/L]"] = (
    y.to_numpy() - y_pred
)

# Compare nearby harvest times for the same operating conditions
candidate_times = [64, 72, 80]


candidate_conditions = df_optimization[
    (
        df_optimization["Initial Glucose [mM]"]
        == optimal_conditions["Initial Glucose [mM]"]
    )
    & (
        df_optimization["Initial Glutamine [mM]"]
        == optimal_conditions["Initial Glutamine [mM]"]
    )
    & (
        df_optimization["Initial VCD [cells/mL]"]
        == optimal_conditions["Initial VCD [cells/mL]"]
    )
    & (
        df_optimization["Time [h]"].isin(candidate_times)
    )
].sort_values("Time [h]").copy()


candidate_times = candidate_conditions["Time [h]"].to_numpy()

predicted_mab = candidate_conditions[
    "Predicted mAb [mg/L]"
].to_numpy()

culture_cost = candidate_conditions[
    "Cell Culture Cost [$/L]"
].to_numpy()

cycle_days = (
    candidate_conditions["Time [d]"].to_numpy() + 1
)

# Create a residual trajectory for each batch
residual_matrix = df_residuals[
    df_residuals["Time [h]"].isin(candidate_times)
].pivot(
    index="Batch ID",
    columns="Time [h]",
    values="Residual [mg/L]"
).reindex(columns=candidate_times)

# Only retain batches with all three time points
residual_matrix = residual_matrix.dropna()

if residual_matrix.empty:
    raise ValueError(
        "No complete batch residual trajectories found."
    )


# Extract residual trajectories for each batch
residual_values = residual_matrix.to_numpy()

# Center residuals independently at each harvest time
residual_values = (
    residual_values
    - residual_values.mean(axis=0, keepdims=True)
)

# Verify that the residual means are approximately zero
print(
    "\nCentered residual means:",
    residual_values.mean(axis=0)
)


# Sample entire batch residual trajectories
sample_indices = rng.integers(
    0,
    len(residual_values),
    size=n_simulations
)

sampled_residuals = residual_values[sample_indices]

# Simulate concentrations for the candidate conditions
simulated_mab = np.maximum(
    predicted_mab[None, :] + sampled_residuals,
    0
)

simulated_profit = (
    1000 * simulated_mab - culture_cost[None, :]
) / cycle_days[None, :]

# Which candidate has the highest profit in each simulation?
winning_indices = np.argmax(simulated_profit, axis=1)

winning_times = candidate_times[winning_indices]

robustness_results = []

for i, time in enumerate(candidate_times):

    profits = simulated_profit[:, i]

    robustness_results.append({
        "Harvest Time [h]": time,
        "Mean Simulated Profit [$/L/d]": profits.mean(),
        "Profit 2.5th Percentile": np.percentile(profits, 2.5),
        "Profit 97.5th Percentile": np.percentile(profits, 97.5),
        "Probability of Highest Profit":
            np.mean(winning_times == time)
    })

df_robustness = pd.DataFrame(robustness_results)

print("\nOPTIMIZATION ROBUSTNESS ANALYSIS")
print(df_robustness.round(4).to_string(index=False))

df_robustness.to_csv(
    os.path.join("tables", "optimization_robustness.csv"),
    index=False
)

# =============================================================================
# ACTUAL VS PREDICTED
# =============================================================================

plt.figure(figsize=(6, 4))

plt.scatter(
    y,
    y_pred,
    alpha=0.4
)

limits = [
    min(y.min(), y_pred.min()),
    max(y.max(), y_pred.max())
]

plt.plot(
    limits,
    limits,
    "r--"
)

plt.xlabel("Actual mAb [mg/L]")
plt.ylabel("Predicted mAb [mg/L]")
plt.title("Cross-Validated Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    os.path.join("figures", "actual_vs_predicted.png"),
    dpi=300
)

plt.close()

# =============================================================================
# RESIDUAL PLOT
# =============================================================================

residuals = y - y_pred

plt.figure(figsize=(6, 4))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.4
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted mAb [mg/L]")
plt.ylabel("Residual [mg/L]")
plt.title("Cross-Validated Residuals")

plt.tight_layout()

plt.savefig(
    os.path.join("figures", "residual_plot.png"),
    dpi=300
)

plt.close()

print(df_combined["mAb [mg/L]"].describe())

# =============================================================================
# PREDICTION ERROR VS TIME
# =============================================================================

df_validation = df_combined.copy()

df_validation["Predicted mAb [mg/L]"] = y_pred

df_validation["Absolute Error [mg/L]"] = np.abs(
    df_validation["mAb [mg/L]"]
    - df_validation["Predicted mAb [mg/L]"]
)

df_time_error = (
    df_validation
    .groupby("Time [d]")["Absolute Error [mg/L]"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

plt.plot(
    df_time_error["Time [d]"],
    df_time_error["Absolute Error [mg/L]"]
)

plt.axvline(
    optimal_conditions["Time [d]"],
    color="red",
    linestyle="--",
    label="Optimal Harvest Time"
)

plt.xlabel("Time [d]")
plt.ylabel("Mean Absolute Error [mg/L]")
plt.title("Prediction Error vs Harvest Time")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "prediction_error_vs_time.png"),
    dpi=300
)

plt.close()

# =============================================================================
# APPROXIMATE 95% PREDICTION INTERVALS
# =============================================================================

# Select residuals near the recommended harvest time
optimal_time = optimal_conditions["Time [d]"]

time_mask = (
    np.abs(df_combined["Time [d]"] - optimal_time)
    <= 0.5
)

residuals_near_optimum = residuals[time_mask]

# Empirical residual quantiles
lower_residual = np.percentile(
    residuals_near_optimum, 2.5
)

upper_residual = np.percentile(
    residuals_near_optimum, 97.5
)

# Predicted mAb concentration
predicted_mab = optimal_conditions["Predicted mAb [mg/L]"]

# Approximate prediction interval
mab_lower = predicted_mab + lower_residual
mab_upper = predicted_mab + upper_residual

# Culture cost
culture_cost = optimal_conditions["Cell Culture Cost [$/L]"]

# Culture value interval
value_lower = 1000 * mab_lower
value_upper = 1000 * mab_upper

# Space-time profit interval
profit_lower = (
    value_lower - culture_cost
) / (optimal_time + 1)

profit_upper = (
    value_upper - culture_cost
) / (optimal_time + 1)

# Space-time yield interval
yield_lower = mab_lower / (optimal_time + 1)
yield_upper = mab_upper / (optimal_time + 1)

interval_summary = pd.DataFrame({
    "Performance Metric": [
        "mAb Concentration [mg/L]",
        "Culture Value [$/L]",
        "Space-Time Profit [$/L/d]",
        "Space-Time Yield [mg/L/d]"
    ],
    "Predicted Value": [
        predicted_mab,
        1000 * predicted_mab,
        optimal_conditions["Predicted Space-Time Profit [$/L/d]"],
        optimal_conditions["Predicted Space-Time Yield [mg/L/d]"]
    ],
    "Lower 95% Bound": [
        mab_lower,
        value_lower,
        profit_lower,
        yield_lower
    ],
    "Upper 95% Bound": [
        mab_upper,
        value_upper,
        profit_upper,
        yield_upper
    ]
})

interval_summary_rounded = interval_summary.round({
    "Predicted Value": 4,
    "Lower 95% Bound": 4,
    "Upper 95% Bound": 4
})

interval_summary_rounded.to_csv(
    os.path.join("tables", "prediction_interval_summary.csv"),
    index=False
)

print("\nAPPROXIMATE 95% PREDICTION INTERVALS")
print("=" * 45)

print(
    f"mAb Concentration: "
    f"[{mab_lower:.4f}, {mab_upper:.4f}] mg/L"
)

print(
    f"Cell Culture Value: "
    f"[${value_lower:.2f}, ${value_upper:.2f}]/L"
)

print(
    f"Space-Time Profit: "
    f"[${profit_lower:.2f}, ${profit_upper:.2f}]/L/day"
)

print(
    f"Space-Time Yield: "
    f"[{yield_lower:.4f}, {yield_upper:.4f}] mg/L/day"
)

# =============================================================================
# SPACE-TIME PROFIT VS HARVEST TIME
# =============================================================================

df_harvest = df_optimization[
    (df_optimization["Initial Glucose [mM]"] == optimal_conditions["Initial Glucose [mM]"])
    & (df_optimization["Initial Glutamine [mM]"] == optimal_conditions["Initial Glutamine [mM]"])
    & (df_optimization["Initial VCD [cells/mL]"] == optimal_conditions["Initial VCD [cells/mL]"])
].sort_values("Time [d]")

plt.figure(figsize=(8, 5))

plt.plot(
    df_harvest["Time [d]"],
    df_harvest["Predicted Space-Time Profit [$/L/d]"],
    label="Predicted Space-Time Profit"
)

plt.axvline(
    optimal_conditions["Time [d]"],
    color="red",
    linestyle="--",
    label="Optimal Harvest Time"
)

plt.xlabel("Harvest Time [d]")
plt.ylabel("Space-Time Profit [$/L/day]")
plt.title("Effect of Harvest Time on Space-Time Profit")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "profit_vs_harvest_time.png"),
    dpi=300
)

plt.close()

# =============================================================================
# GLUCOSE-GLUTAMINE PROFIT HEATMAP
# =============================================================================

df_heatmap = df_optimization[
    (df_optimization["Initial VCD [cells/mL]"] == optimal_conditions["Initial VCD [cells/mL]"])
    & (np.isclose(
    df_optimization["Time [d]"],
    optimal_conditions["Time [d]"]
))
]

profit_matrix = df_heatmap.pivot(
    index="Initial Glutamine [mM]",
    columns="Initial Glucose [mM]",
    values="Predicted Space-Time Profit [$/L/d]"
)

plt.figure(figsize=(8, 6))

plt.imshow(
    profit_matrix,
    origin="lower",
    aspect="auto",
    extent=[
        profit_matrix.columns.min(),
        profit_matrix.columns.max(),
        profit_matrix.index.min(),
        profit_matrix.index.max()
    ]
)

plt.colorbar(label="Space-Time Profit [$/L/day]")

plt.scatter(
    optimal_conditions["Initial Glucose [mM]"],
    optimal_conditions["Initial Glutamine [mM]"],
    color="red",
    marker="x",
    s=100,
    label="Optimal Conditions"
)

plt.xlabel("Initial Glucose [mM]")
plt.ylabel("Initial Glutamine [mM]")
plt.title("Effect of Initial Nutrients on Space-Time Profit")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "profit_heatmap.png"),
    dpi=300
)

plt.close()

# =============================================================================
# VCD SENSITIVITY ANALYSIS
# =============================================================================

df_vcd = df_optimization[
    (df_optimization["Initial Glucose [mM]"] == optimal_conditions["Initial Glucose [mM]"])
    & (df_optimization["Initial Glutamine [mM]"] == optimal_conditions["Initial Glutamine [mM]"])
    & (np.isclose(
    df_optimization["Time [d]"],
    optimal_conditions["Time [d]"]
))
].sort_values("Initial VCD [cells/mL]")

plt.figure(figsize=(8, 5))

plt.plot(
    df_vcd["Initial VCD [cells/mL]"],
    df_vcd["Predicted Space-Time Profit [$/L/d]"],
    marker="o",
    label="Predicted Space-Time Profit"
)

plt.axvline(
    optimal_conditions["Initial VCD [cells/mL]"],
    color="red",
    linestyle="--",
    label="Optimal Initial VCD"
)

plt.xlabel("Initial VCD [cells/mL]")
plt.ylabel("Space-Time Profit [$/L/day]")
plt.title("Effect of Initial VCD on Space-Time Profit")

plt.ticklabel_format(
    axis="x",
    style="sci",
    scilimits=(6, 6)
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "profit_vs_vcd.png"),
    dpi=300
)

plt.close()

# =============================================================================
# VCD SENSITIVITY: SPACE-TIME YIELD
# =============================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    df_vcd["Initial VCD [cells/mL]"],
    df_vcd["Predicted Space-Time Yield [mg/L/d]"],
    marker="o",
    label="Predicted Space-Time Yield"
)

plt.axvline(
    optimal_conditions["Initial VCD [cells/mL]"],
    color="red",
    linestyle="--",
    label="Profit-Optimal Initial VCD"
)

plt.xlabel("Initial VCD [cells/mL]")
plt.ylabel("Space-Time Yield [mg/L/day]")
plt.title("Effect of Initial VCD on Space-Time Yield")

plt.ticklabel_format(
    axis="x",
    style="sci",
    scilimits=(6, 6)
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "yield_vs_vcd.png"),
    dpi=300
)

plt.close()

# =============================================================================
# COMPARE PROFIT AND YIELD OPTIMA
# =============================================================================

profit_optimum = df_optimization.loc[
    df_optimization[
        "Predicted Space-Time Profit [$/L/d]"
    ].idxmax()
]

yield_optimum = df_optimization.loc[
    df_optimization[
        "Predicted Space-Time Yield [mg/L/d]"
    ].idxmax()
]

comparison_columns = [
    "Initial Glucose [mM]",
    "Initial Glutamine [mM]",
    "Initial VCD [cells/mL]",
    "Time [d]",
    "Predicted Space-Time Profit [$/L/d]",
    "Predicted Space-Time Yield [mg/L/d]"
]

df_optima = pd.DataFrame(
    [profit_optimum, yield_optimum],
    index=["Profit Optimum", "Yield Optimum"]
)

print("\nPROFIT VS YIELD OPTIMIZATION")
optima_rounding = {
    "Initial Glucose [mM]": 1,
    "Initial Glutamine [mM]": 1,
    "Initial VCD [cells/mL]": 0,
    "Time [d]": 2,
    "Raw Predicted mAb [mg/L]": 4,
    "Predicted mAb [mg/L]": 4,
    "Predicted Cell Culture Value [$/L]":4,
    "Predicted Profit [$/L]": 4,
    "Predicted Space-Time Profit [$/L/d]": 2,
    "Predicted Space-Time Yield [mg/L/d]": 4,
}

df_optima_rounded = df_optima.round(optima_rounding)

print("\nPROFIT VS. YIELD OPTIMA")
print(df_optima_rounded.to_string(index=False))

df_optima_rounded.to_csv(os.path.join("tables","profit_vs_yield_optima.csv"),index=False)

# =============================================================================
# EARLY-HARVEST SENSITIVITY ANALYSIS
# =============================================================================

minimum_harvest_times = [8, 16, 24, 48, 72]

sensitivity_results = []

for minimum_time in minimum_harvest_times:

    eligible_conditions = df_optimization[
        df_optimization["Time [h]"] >= minimum_time
    ]

    profit_index = eligible_conditions[
        "Predicted Space-Time Profit [$/L/d]"
    ].idxmax()

    yield_index = eligible_conditions[
        "Predicted Space-Time Yield [mg/L/d]"
    ].idxmax()

    profit_opt = eligible_conditions.loc[profit_index]
    yield_opt = eligible_conditions.loc[yield_index]

    sensitivity_results.append({
        "Minimum Harvest Time [h]": minimum_time,
        "Profit Optimal Time [h]": profit_opt["Time [h]"],
        "Profit Optimal Glucose [mM]": profit_opt["Initial Glucose [mM]"],
        "Profit Optimal Glutamine [mM]": profit_opt["Initial Glutamine [mM]"],
        "Profit Optimal VCD [cells/mL]": profit_opt["Initial VCD [cells/mL]"],
        "Maximum Space-Time Profit [$/L/d]":
            profit_opt["Predicted Space-Time Profit [$/L/d]"],
        "Yield Optimal Time [h]": yield_opt["Time [h]"],
        "Maximum Space-Time Yield [mg/L/d]":
            yield_opt["Predicted Space-Time Yield [mg/L/d]"]
    })

df_sensitivity = pd.DataFrame(sensitivity_results)

print("\nEARLY-HARVEST SENSITIVITY ANALYSIS")
sensitivity_rounding = {
    "Minimum Harvest Time [h]": 0,
    "Profit Optimal Time [h]": 0,
    "Profit Optimal Glucose [mM]": 1,
    "Profit Optimal Glutamine [mM]": 1,
    "Profit Optimal VCD [cells/mL]": 0,
    "Maximum Space-Time Profit [$/L/d]": 2,
    "Yield Optimal Time [h]": 0,
    "Maximum Space-Time Yield [mg/L/d]": 4,
}

df_sensitivity_rounded = df_sensitivity.round(sensitivity_rounding)

print(df_sensitivity_rounded.to_string(index=False))

df_sensitivity_rounded.to_csv(os.path.join("tables","early_harvest_sensitivity.csv"), index=False)

# =============================================================================
# EARLY-TIME MODEL VALIDATION
# =============================================================================

df_early = df_combined.copy()

y_pred_cv = cross_val_predict(
    best_model,
    X,
    y,
    cv=GroupKFold(n_splits=5),
    groups=groups,
    n_jobs=-1
)

df_early["Predicted mAb [mg/L]"] = y_pred_cv

df_early["Absolute Error [mg/L]"] = abs(
    df_early["mAb [mg/L]"]
    - df_early["Predicted mAb [mg/L]"]
)

early_results = (
    df_early[
        df_early["Time [h]"].isin([0, 8, 16, 24, 48, 72])
    ]
    .groupby("Time [h]")
    .agg(
        Mean_Actual=("mAb [mg/L]", "mean"),
        Mean_Predicted=("Predicted mAb [mg/L]", "mean"),
        Mean_Absolute_Error=("Absolute Error [mg/L]", "mean"),
        Maximum_Absolute_Error=("Absolute Error [mg/L]", "max")
    )
    .round(4)
)

print("\nEARLY-TIME PREDICTION VALIDATION")
print(early_results.to_string())

# =============================================================================
# EARLY-TIME PREDICTION BIAS
# =============================================================================

early_results["Prediction_Bias"] = (
    early_results["Mean_Predicted"]
    - early_results["Mean_Actual"]
)

plt.figure(figsize=(9, 5))

plt.plot(
    early_results.index,
    early_results["Prediction_Bias"],
    marker="o",
    linewidth=2,
    label="Mean Prediction Bias"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    label="Zero Bias"
)

plt.xlabel("Harvest Time [h]")
plt.ylabel("Prediction Bias [mg/L]")
plt.title("Early-Time Prediction Bias")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join("figures","early_time_prediction_bias.png"),dpi=300)

plt.close()