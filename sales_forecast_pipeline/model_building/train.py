# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("superkart-sales-training-experiment")

api = HfApi()

data_repo_id = "Jganesh045/superkart-sales-data"
Xtrain_path = f"hf://datasets/{data_repo_id}/Xtrain.csv"
Xtest_path = f"hf://datasets/{data_repo_id}/Xtest.csv"
ytrain_path = f"hf://datasets/{data_repo_id}/ytrain.csv"
ytest_path = f"hf://datasets/{data_repo_id}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define numeric and categorical features
numeric_features = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
categorical_features = ["Product_Sugar_Content", "Product_Type", "Store_Size",
                         "Store_Location_City_Type", "Store_Type"]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base model -- Random Forest was the strongest candidate in the development-stage
# comparison (see notebook cells above): it beat XGBoost, Gradient Boosting, Bagging,
# a single Decision Tree, and AdaBoost on 5-fold CV RMSE for this dataset.
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# Hyperparameter grid
param_grid = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_depth": [10, 20],
    "randomforestregressor__min_samples_leaf": [1, 2],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1,
                                scoring="neg_root_mean_squared_error")
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_rmse", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics (sqrt(MSE) rather than squared=False -- that argument was removed in
    # newer scikit-learn versions, so this form is compatible across versions)
    train_rmse = mean_squared_error(ytrain, y_pred_train) ** 0.5
    test_rmse = mean_squared_error(ytest, y_pred_test) ** 0.5
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    test_mae = mean_absolute_error(ytest, y_pred_test)
    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_RMSE": train_rmse, "test_RMSE": test_rmse,
        "train_MAE": train_mae, "test_MAE": test_mae,
        "train_R2": train_r2, "test_R2": test_r2
    })

    # Save the model locally
    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Jganesh045/superkart-sales-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_superkart_sales_model_v1.joblib",
        path_in_repo="best_superkart_sales_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Model registered to Hugging Face Hub.")
