from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="sales_forecast_pipeline/deployment",
    repo_id="Jganesh045/superkart-sales-forecast",
    repo_type="space",
    path_in_repo="",
)
