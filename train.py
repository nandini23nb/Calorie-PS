import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from preprocess import load_data, engineer_features, preprocess

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
EX_PATH      = os.path.join(DATA_DIR, "exercise.csv")
CAL_PATH     = os.path.join(DATA_DIR, "calories.csv")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── 1. EDA ──────────────────────────────────────────────────────────────────
def run_eda():
    print("\n📊 Running Exploratory Data Analysis...")
    df = load_data(EX_PATH, CAL_PATH)
    df = engineer_features(df)

    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Calorie Prediction — EDA Dashboard", fontsize=16, fontweight="bold", y=1.01)

    # 1. Calories distribution
    axes[0, 0].hist(df["Calories"], bins=40, color="#e8ff47", edgecolor="#111", alpha=0.85)
    axes[0, 0].set_title("Calories Distribution")
    axes[0, 0].set_xlabel("Calories Burned (kcal)")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Calories by gender
    df["Gender_Label"] = df["Gender"]
    sns.boxplot(data=df, x="Gender_Label", y="Calories", ax=axes[0, 1],
                palette={"male": "#4df0ff", "female": "#ff6b35"})
    axes[0, 1].set_title("Calories by Gender")

    # 3. Duration vs Calories
    axes[0, 2].scatter(df["Duration"], df["Calories"], alpha=0.3, s=8, color="#e8ff47")
    axes[0, 2].set_title("Duration vs Calories")
    axes[0, 2].set_xlabel("Duration (min)")
    axes[0, 2].set_ylabel("Calories (kcal)")

    # 4. Heart Rate vs Calories
    axes[1, 0].scatter(df["Heart_Rate"], df["Calories"], alpha=0.3, s=8, color="#ff6b35")
    axes[1, 0].set_title("Heart Rate vs Calories")
    axes[1, 0].set_xlabel("Heart Rate (bpm)")

    # 5. Correlation heatmap
    corr_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI", "Calories"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=axes[1, 1], linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[1, 1].set_title("Correlation Heatmap")

    # 6. Age vs Calories
    axes[1, 2].scatter(df["Age"], df["Calories"], alpha=0.3, s=8, color="#a78bfa")
    axes[1, 2].set_title("Age vs Calories")
    axes[1, 2].set_xlabel("Age (years)")

    plt.tight_layout()
    eda_path = os.path.join(MODELS_DIR, "eda_plots.png")
    plt.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ EDA plots saved → {eda_path}")

    # Print summary stats
    print("\n   Dataset Shape:", load_data(EX_PATH, CAL_PATH).shape)
    print("   Calories stats:")
    print(df["Calories"].describe().to_string())


# ─── 2. Train ────────────────────────────────────────────────────────────────
def train_models():
    print("\n🤖 Training Models...")
    X, y, le, _, feature_names = preprocess(EX_PATH, CAL_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        ),
    }

    results = {}
    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # 5-fold CV
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)

        results[name] = {
            "model": model,
            "R²": r2,
            "MAE": mae,
            "RMSE": rmse,
            "CV_R²_mean": cv_scores.mean(),
            "CV_R²_std": cv_scores.std(),
        }

        print(f"   → R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  CV_R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = name

    print(f"\n   🏆 Best model: {best_model} (R²={best_r2:.4f})")
    return models, results, best_model, le, feature_names, X_test, y_test


# ─── 3. Plots ─────────────────────────────────────────────────────────────────
def plot_results(models, results, best_model, feature_names, X_test, y_test):
    print("\n📈 Generating result plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Evaluation Dashboard", fontsize=15, fontweight="bold")

    # Actual vs Predicted (best model)
    best = models[best_model]
    preds = best.predict(X_test)
    axes[0].scatter(y_test, preds, alpha=0.3, s=10, color="#4df0ff")
    lim = [0, y_test.max() + 10]
    axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Calories")
    axes[0].set_ylabel("Predicted Calories")
    axes[0].set_title(f"Actual vs Predicted\n{best_model}")
    axes[0].legend()

    # Model comparison bar chart
    model_names = list(results.keys())
    r2_vals = [results[n]["R²"] for n in model_names]
    colors = ["#e8ff47" if n == best_model else "#2a2f3d" for n in model_names]
    bars = axes[1].bar(model_names, r2_vals, color=colors, edgecolor="#555")
    axes[1].set_ylim(0.99, 1.001)
    axes[1].set_title("Model R² Comparison")
    axes[1].set_ylabel("R² Score")
    for bar, val in zip(bars, r2_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    # Feature importance (best model)
    if hasattr(best, "feature_importances_"):
        fi = pd.Series(best.feature_importances_, index=feature_names).sort_values(ascending=True)
        fi.plot(kind="barh", ax=axes[2], color="#ff6b35")
        axes[2].set_title(f"Feature Importance\n{best_model}")
        axes[2].set_xlabel("Importance Score")

    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "model_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Result plots saved → {plot_path}")


# ─── 4. Save ─────────────────────────────────────────────────────────────────
def save_artifacts(models, results, best_model, le, feature_names):
    print("\n💾 Saving model artifacts...")
    artifacts = {
        "models": {n: results[n]["model"] for n in results},
        "best_model_name": best_model,
        "best_model": models[best_model],
        "label_encoder": le,
        "feature_names": feature_names,
        "metrics": {
            n: {k: v for k, v in results[n].items() if k != "model"}
            for n in results
        },
    }
    save_path = os.path.join(MODELS_DIR, "calorie_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"   ✅ Artifacts saved → {save_path}")
    return save_path


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CalorieAI — Model Training Pipeline")
    print("=" * 60)

    run_eda()
    models, results, best_model, le, feature_names, X_test, y_test = train_models()
    plot_results(models, results, best_model, feature_names, X_test, y_test)
    save_artifacts(models, results, best_model, le, feature_names)

    print("\n✅ Training complete! Run: streamlit run app.py")
    print("=" * 60)
