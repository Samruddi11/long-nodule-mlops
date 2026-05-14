import mlflow

mlflow.set_experiment("Lung Nodule Detection using Radiomics")

with mlflow.start_run():
    mlflow.log_param("model_type", "PyTorch")
    mlflow.log_param("model_file", "model.pt")
    mlflow.log_param("dataset_type", "radiomics_xlsx")

    mlflow.log_metric("accuracy", 0.9375)
    mlflow.log_metric("loss", 0.1989)

    mlflow.log_artifact("trained_model/model.pt")

print("MLflow tracking completed successfully")