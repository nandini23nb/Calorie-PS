# 🔥 CalorieAI — AI-Powered Calorie Prediction System

Final Year Project | Machine Learning | Regression Analysis

---

## 📌 Problem Statement
Build a regression-based ML model to predict calories burned during physical activity using physiological and activity-related features.

## 🗂️ Project Structure

```
calories_ai/
├── data/
│   ├── exercise.csv        # 15,000 records — physiological features
│   └── calories.csv        # 15,000 records — calorie labels
├── models/
│   ├── calorie_model.pkl   # Saved model artifacts (after training)
│   ├── eda_plots.png       # EDA dashboard plot
│   └── model_results.png   # Model evaluation plots
├── preprocess.py           # Feature engineering & preprocessing pipeline
├── train.py                # Model training, evaluation, and saving
├── app.py                  # Streamlit web application
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This will:
- Run EDA and save plots
- Train 3 models: Random Forest, Gradient Boosting, XGBoost
- Evaluate with R², MAE, RMSE, and 5-fold Cross Validation
- Save the best model to `models/calorie_model.pkl`

### 3. Launch the web app
```bash
streamlit run app.py
```

---

## 🔬 Features Used

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Age | Numeric | 20–79 years |
| Height | Numeric | cm |
| Weight | Numeric | kg |
| Duration | Numeric | Exercise minutes |
| Heart Rate | Numeric | bpm |
| Body Temperature | Numeric | °C |
| **BMI** | Engineered | Weight / Height² |
| **HR Intensity** | Engineered | Heart Rate / (220 - Age) |
| **Duration × HR** | Engineered | Interaction term |
| **Temp Deviation** | Engineered | Body Temp - 37°C |

## 📊 Model Performance

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Random Forest | 0.9982 | 1.72 | 2.68 |
| Gradient Boosting | ~0.998 | ~1.8 | ~2.9 |
| **XGBoost** ✅ | **0.9989** | **1.50** | **2.14** |

## 🎯 App Pages
1. **Home & Predict** — Single prediction with feature importance
2. **EDA Dashboard** — Dataset visualizations
3. **Model Performance** — Metrics, comparison, feature importance
4. **Batch Predict** — Upload CSV, download predictions

---

*Powered by XGBoost + Streamlit | Dataset: 15,000 exercise records*
