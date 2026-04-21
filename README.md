# Iris Prediction API

A machine learning operations (MLOps) project for training and deploying an Iris species classification model using MLflow for experiment tracking and FastAPI for REST API endpoints.

## Project Overview

This project demonstrates a complete machine learning pipeline with the following components:

- **Model Training**: Trains Logistic Regression models on the Iris dataset
- **Experiment Tracking**: Uses MLflow to record and track all model parameters, metrics, and artifacts
- **Model Registry**: Registers trained models for easy deployment and version management
- **API Deployment**: Provides a REST API endpoint to make predictions on new data
- **Multi-Algorithm Comparison**: Tests multiple solver algorithms (lbfgs, saga, newton-cg) to compare results

## Project Structure

```
mlops-prediction-api/
├── app.py                 # FastAPI application for predictions
├── train.py              # Model training script with MLflow tracking
├── README.md             # This file
├── mlruns/               # MLflow experiment runs (auto-generated)
└── mlartifacts/          # MLflow model artifacts (auto-generated)
```

## Quick Start

### 1. Create Virtual Environment

```powershell
cd C:\Users\amrit\Desktop\local ML
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Dependencies

```powershell
pip install mlflow scikit-learn fastapi uvicorn --disable-pip-version-check
```

All packages will be cached for faster installation in the future.

### 3. Start MLflow Tracking Server

Open a terminal and run:

```powershell
mlflow server --port 5000
```

Open http://localhost:5000 in your browser to view the MLflow interface.

### 4. Run Training Script

Open another terminal with the virtual environment activated:

```powershell
python train.py
```

This trains a Logistic Regression model and logs it to MLflow.

### 5. Start the Prediction API

Open a third terminal with the virtual environment activated:

```powershell
uvicorn app:app --reload --port 8000
```

The API will be available at http://localhost:8000

## MLflow Experiment Tracking

### View Experiments

Open http://localhost:5000 in your browser and click Training runs in the left sidebar.

### Key Metrics Logged

- `training_accuracy_score` - Model accuracy on training data
- `training_precision_score` - Precision metric
- `training_recall_score` - Recall metric
- `training_f1_score` - F1 score
- `test_accuracy` - Accuracy on test set

### Run Naming

MLflow automatically generates unique run names (for example: "rogue-loon-201", "vaunted-cub-534") to identify each experiment. Each run is automatically named with an adjective, an animal name, and a number.

### Compare Runs

1. Select multiple completed runs using checkboxes
2. Click the "Compare" button
3. View side-by-side comparison of metrics and parameters

## Model Registry

Trained models are registered in the MLflow Model Registry under the name iris-classifier. This creates an official record of your model that can be easily referenced for deployment and production use.

### View Registered Models

In the MLflow UI, navigate to the Models tab to see:
- Model name: iris-classifier
- Version numbers (v1, v2, etc.)
- Latest model information

### Load Registered Model

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/iris-classifier/latest")
```

## API Usage

### Endpoint: /predict

**Method**: POST

**Request Body**:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response**:
```json
{
  "prediction": 0,
  "species": "setosa"
}
```

### Example Predictions

**Test 1 - Setosa** (Short petals):
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```

**Response**: `{"prediction": 0, "species": "setosa"}`

**Test 2 - Virginica** (Long petals):
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 6.7,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.3
}'
```

**Response**: `{"prediction": 2, "species": "virginica"}`

### Interactive API Documentation

Swagger UI: http://localhost:8000/docs  
ReDoc: http://localhost:8000/redoc

## Training Script Configuration

Edit train.py to modify hyperparameters:

```python
# You can change the solver algorithm and other parameters
# Different solver options available: lbfgs, saga, newton-cg
params = {"solver": "lbfgs", "max_iter": 1000, "random_state": 8888}
# params = {"solver": "saga", "max_iter": 1000, "random_state": 8888}
# params = {"solver": "newton-cg", "max_iter": 200, "random_state": 8888}
```

### Supported Solvers

- **lbfgs**: Limited-memory BFGS algorithm, works well for small datasets
- **saga**: SAGA solver, efficient for large datasets
- **newton-cg**: Newton Conjugate Gradient, a robust optimization method

Run train.py multiple times with different solvers to compare performance in the MLflow UI.

## Model Performance

Typical results from the Iris dataset:

| Solver | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| lbfgs  | 97.5%    | 97.67%    | 97.5%  | 97.49%   |
| saga   | ~97%     | ~97%      | ~97%   | ~97%     |
| newton-cg | ~97%  | ~97%      | ~97%   | ~97%     |

## MLflow Architecture

```
train.py
   ↓
mlflow.set_tracking_uri("http://localhost:5000")
   ↓
MLflow Server (Port 5000)
   ↓
├── Experiment Registry
├── Run Metrics & Parameters
├── Model Artifacts
└── Model Registry
   ↓
app.py (loads latest model)
   ↓
FastAPI (Port 8000)
```

## Dependencies

- **mlflow**: Experiment tracking and model registry
- **scikit-learn**: Machine learning algorithms
- **fastapi**: Web API framework
- **uvicorn**: ASGI server
- **numpy, pandas, scipy**: Data processing
- **matplotlib**: Plotting

## Troubleshooting

### Virtual Environment Issues

**Problem**: ./activate command is not recognized

**Solution**: In PowerShell, use the correct activation script path:
```powershell
.\venv\Scripts\Activate.ps1
```

### Installation Timeout

**Problem**: Installation hangs or takes a very long time

**Solution**: Use the cached packages flag to speed up installation:
```powershell
pip install mlflow scikit-learn fastapi uvicorn --disable-pip-version-check
```

### MLflow UI Shows No Experiments

**Problem**: When you open the MLflow UI, no experiments appear

**Solution**: Verify that train.py includes the tracking URI to connect to the server:
```python
mlflow.set_tracking_uri("http://localhost:5000")
```

### API Returns No Predictions

**Problem**: The /predict endpoint does not return predictions

**Solution**: 
1. Verify the MLflow server is running on port 5000
2. Check that your model is registered at http://localhost:5000/#/models
3. Restart app.py if the model registry was recently updated

## Next Steps

1. Train models with different solvers
2. Compare experiments in MLflow UI
3. Register best model
4. Test API predictions
5. Deploy to production (Docker, cloud platforms)
6. Monitor model performance over time
7. Set up automated retraining pipelines

## Resources

- [MLflow Documentation](https://mlflow.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Iris Dataset Info](https://en.wikipedia.org/wiki/Iris_flower_data_set)

## Author

Amrit

## License

MIT
