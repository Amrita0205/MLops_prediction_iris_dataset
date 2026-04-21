import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset and split into training and test sets
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Point MLflow to the tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Tell MLflow which experiment to log this run under
mlflow.set_experiment("iris-classification")

# Enable automatic logging for scikit-learn models
mlflow.sklearn.autolog()

# Define the hyperparameters for the model (we can change and check these hyper parameters as well)
# params = {"solver": "lbfgs", "max_iter": 1000, "random_state": 8888}
# params = {"solver": "saga", "max_iter": 1000, "random_state": 8888}
params = {"solver": "newton-cg", "max_iter": 200, "random_state": 8888}

# Start an MLflow run to track everything
with mlflow.start_run() as run:  # ← add "as run" here
    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Manually log the test accuracy metric
    mlflow.log_metric("test_accuracy", accuracy)

    # Save the trained model as an artifact
    mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    # Add a descriptive tag to this run
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    print(f"Test accuracy: {accuracy:.4f}")

    # Register the model in the MLflow Model Registry  ← ADD THIS
    model_uri = f"runs:/{run.info.run_id}/iris_model"
    mlflow.register_model(model_uri, "iris-classifier")