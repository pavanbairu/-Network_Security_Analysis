import os
import sys
import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.estimator import NetworkModel
from networksecurity.utils.common import load_object
from networksecurity.constant.training_pipeline import FINAL_PREPROCESSOR_PATH, FINAL_MODEL_PATH, FINAL_PREDICTED_OUTPUT

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

# Initialize FastAPI application
app = FastAPI()

# Allow Cross-Origin Resource Sharing (CORS) from all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    """
    Redirects the user to the API documentation page.

    Args:
        None

    Returns:
        RedirectResponse: Redirects to the "/docs" page.
    """
    return RedirectResponse("/docs")


@app.get("/train")
async def train():
    """
    Starts the training pipeline to train a machine learning model.

    Args:
        None

    Returns:
        Response: Returns a success message if training is successful.
    """
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Handles prediction for a given CSV file:
    - Reads the uploaded file.
    - Preprocesses the data using a pre-trained preprocessor.
    - Uses a trained model to make predictions.
    - Returns predictions as an HTML table.

    Args:
        request (Request): FastAPI request object.
        file (UploadFile): Uploaded CSV file containing input data for predictions.

    Returns:
        TemplateResponse: Renders an HTML page with the predictions in a table format.
    """
    try:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(file.file)

        # Load pre-trained preprocessor and model
        preprocessor = load_object(FINAL_PREPROCESSOR_PATH)
        final_model = load_object(FINAL_MODEL_PATH)

        # Initialize the model with preprocessor
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Perform predictions
        y_pred = network_model.predict(df)

        # Add predictions to the DataFrame
        df['predicted_output'] = y_pred

        # Save predictions to a CSV file
        df.to_csv(FINAL_PREDICTED_OUTPUT, index=False)

        # Render predictions as an HTML table
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    """
    Runs the FastAPI application on the specified host and port.
    """
    run(app, host="0.0.0.0", port=8000)
