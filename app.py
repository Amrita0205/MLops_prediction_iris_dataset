from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

# Define the expected shape of incoming prediction requests
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create the FastAPI app instance
app = FastAPI(title="Iris Prediction API")

# Connect to the MLflow tracking server and load the registered model
mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model("models:/iris-classifier/latest")

# Map numeric predictions to species names
SPECIES = ["setosa", "versicolor", "virginica"]

@app.post("/predict")
def predict(input_data: IrisInput):
    # Convert the input into the format the model expects
    features = [[input_data.sepal_length, input_data.sepal_width,
                 input_data.petal_length, input_data.petal_width]]
    prediction = model.predict(features)
    species = SPECIES[int(prediction[0])]
    return {"prediction": int(prediction[0]), "species": species}