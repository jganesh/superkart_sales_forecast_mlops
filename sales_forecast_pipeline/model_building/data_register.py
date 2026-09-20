from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Jganesh045/superkart-sales-data"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repo already exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Upload the raw data file
api.upload_file(
    path_or_fileobj="sales_forecast_pipeline/data/SuperKart.csv",
    path_in_repo="SuperKart.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Raw dataset uploaded to Hugging Face Hub.")
