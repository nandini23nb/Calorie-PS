"""
preprocess.py
Data loading, cleaning, and feature engineering for the Calorie Prediction System.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(exercise_path: str, calories_path: str) -> pd.DataFrame:
    """Load and merge the exercise and calories datasets."""
    exercise_df = pd.read_csv(exercise_path)
    calories_df = pd.read_csv(calories_path)
    df = exercise_df.merge(calories_df[["User_ID", "Calories"]], on="User_ID")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve model performance."""
    df = df.copy()

    # BMI
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

    # Heart Rate Intensity Ratio: how hard relative to estimated max HR
    df["HR_Intensity"] = df["Heart_Rate"] / (220 - df["Age"])

    # Duration × Heart Rate interaction
    df["Duration_HR"] = df["Duration"] * df["Heart_Rate"]

    # Body temp deviation from normal (37°C)
    df["Temp_Deviation"] = df["Body_Temp"] - 37.0

    return df


def preprocess(
    exercise_path: str,
    calories_path: str,
    fit_encoders: bool = True,
    le: LabelEncoder = None,
    scaler: StandardScaler = None,
):
    """
    Full pipeline: load → engineer → encode → scale.

    Returns
    -------
    X : pd.DataFrame   feature matrix
    y : pd.Series      target (Calories)
    le : LabelEncoder  fitted on Gender
    scaler : StandardScaler  fitted on numeric columns
    feature_names : list[str]
    """
    df = load_data(exercise_path, calories_path)
    df = engineer_features(df)
    df.drop(columns=["User_ID"], inplace=True)

    # Encode gender
    if fit_encoders:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"])  # female=0, male=1
    else:
        df["Gender"] = le.transform(df["Gender"])

    y = df["Calories"]
    X = df.drop(columns=["Calories"])

    feature_names = X.columns.tolist()
    return X, y, le, scaler, feature_names


def preprocess_single(row: dict, le: LabelEncoder) -> pd.DataFrame:
    """
    Preprocess a single input dict for inference.

    Parameters
    ----------
    row : dict with keys: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
    le  : fitted LabelEncoder for Gender

    Returns
    -------
    pd.DataFrame with engineered features, ready for model.predict()
    """
    df = pd.DataFrame([row])
    df = engineer_features(df)
    df["Gender"] = le.transform(df["Gender"])

    # Ensure column order matches training
    feature_cols = [
        "Gender", "Age", "Height", "Weight", "Duration",
        "Heart_Rate", "Body_Temp", "BMI", "HR_Intensity",
        "Duration_HR", "Temp_Deviation",
    ]
    return df[feature_cols]
