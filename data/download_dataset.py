"""
Download the Pima Indians Diabetes Dataset from UCI ML Repository / Kaggle.
Saves it to data/diabetes.csv
"""
import os
import pandas as pd

DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def download():
    output_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return output_path

    print("Downloading Pima Indians Diabetes Dataset...")
    df = pd.read_csv(DATASET_URL, header=None, names=COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} ({len(df)} rows)")
    return output_path


if __name__ == "__main__":
    download()
