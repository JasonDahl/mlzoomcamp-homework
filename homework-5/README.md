### Project Overview

#### Objective
This project focuses on preparing a pre-trained machine learning model for deployment using Flask and containerizing it with Docker. The primary goal is to create a web service that predicts credit probabilities for clients based on a predefined dataset.

#### Key Steps

**Pipenv Installation and Scikit-Learn Version Installation:**  
Pipenv, version 2023.10.3, was installed, and Scikit-Learn version 1.3.1 was set up in a dedicated environment for model preparation.

**Model Preparation and Loading:**  
A pre-trained logistic regression model and a dictionary vectorizer were prepared, saved using Pickle, and made available for download.

**Model Usage:**  
*Scripted Model Loading and Application*  
The notebook includes a script, 'q3_test.py', demonstrating the loading of pre-trained models using Pickle. This script applies the loaded models to estimate credit probabilities for a defined client profile.

**Notable Features:**
- *Pickle-based Model Loading:* Demonstrates how to load pre-trained models ('model1.bin' and 'dv.bin') using Pickle in Python.
- *Client Profile Scoring:* Estimates the credit probability for a specific client profile ('{"job": "retired", "duration": 445, "poutcome": "success"}') using the loaded models.

The script employs error handling techniques for potential issues in unpickling files and reports the calculated credit probability for the defined client profile.

### Web Service Creation

**Flask Web Service Setup:**  
The script 'q4_predict.py' demonstrates the setup of a Flask-based web service. Flask and Gunicorn were installed for serving the model as a web service. The Flask code was written to serve the model, and client credit probabilities were estimated using HTTP requests.

**Important Features:**
- **Model Loading and Utilization:**  
  - Utilizes Pickle to load the previously trained models ('dv.bin' and 'model1.bin').
  - Accepts JSON input for a client profile via HTTP POST requests.
  - Transforms client data using the loaded dictionary vectorizer ('dv') and predicts credit probabilities using the logistic regression model ('model').
  
- **Prediction Endpoint:**
  - Defines an endpoint '/predict' to receive client data and provides predictions.
  - Computes the credit probability and determines if credit will be granted based on a threshold (0.5).
  
- **JSON Response:**
  - Returns a JSON response containing the predicted credit probability and a boolean indicating whether credit will be granted.

The script runs the Flask application with debugging enabled and listens on host '0.0.0.0' at port '9696'.

```
import requests
url = "http://localhost:9696/predict"
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
response = requests.post(url, json=client).json()
print(response)
```

### Docker Integration

Docker was introduced to containerize the Flask-based model serving system. A Dockerfile was created, leveraging a pre-built image, installing dependencies, copying Flask scripts, and running them with Gunicorn.

**Flask Docker Containerization:**
The script 'q6_predict.py' exemplifies the Flask-based web service deployment within a Docker container.

**Important Features:**
- **Model Loading and Utilization:**
  - Uses Pickle to load pre-trained models ('dv.bin' and 'model2.bin').
  - Accepts JSON input for a client profile via HTTP POST requests.
  - Utilizes the loaded dictionary vectorizer ('dv') and logistic regression model ('model') for credit probability prediction.

- **Endpoint for Prediction:**
  - Establishes an endpoint '/predict' to receive client data and provide credit probability predictions.
  - Computes credit probabilities based on the loaded model and dictionary vectorizer.
  
- **JSON Response:**
  - Generates a JSON response containing the estimated credit probability and a boolean indicating whether credit will be granted.

The script initiates the Flask application with debugging enabled, allowing access from any host ('0.0.0.0') on port '9696'.

When running on Localhost, the service can be tested using the 'q6_test.py' script:

```
import requests
url = "http://localhost:9696/predict"
client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url, json=client).json()
print(response)
```

#### Conclusion
The project showcased the steps involved in preparing a pre-trained model using Flask for serving and Docker for containerized deployment, facilitating real-time credit probability estimation.
