# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "Jganesh045/superkart-sales-data"
DATASET_PATH = f"hf://datasets/{repo_id}/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ---- Data cleaning ----
# Standardize inconsistent category label ('reg' -> 'Regular')
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

# Drop identifier columns (not useful for modeling)
df.drop(columns=["Product_Id", "Store_Id"], inplace=True)

# NOTE: categorical columns are intentionally left as raw strings here.
# Encoding (OneHotEncoder) happens inside the training pipeline in train.py so that
# the fitted encoder -- not a separate, disconnected LabelEncoder -- is what's used
# consistently at both training and inference time.

# Define target variable
target_col = "Product_Store_Sales_Total"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="dataset",
    )
print("Train/test splits uploaded to Hugging Face Hub.")
