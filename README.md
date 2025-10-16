# Drug Condition Predictor (Offline)

A **Streamlit app** that predicts medical conditions based on drug names using **sentence-transformers embeddings**.  
The app works **fully offline**, using precomputed embeddings.  
It returns the **Top 5 most similar drugs** and their associated conditions for any query.

## Project Structure
## How to run locally

Follow these steps to clone the repository and run the offline Drug Condition Predictor app.

### Clone the repository

git clone https://github.com/karthiksekaran/disease-prediction-embeddings.git
cd disease-prediction-embeddings

## Create a virtual environment (recommended)

python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

## Install Dependencies

pip install -r requirements.txt
streamlit run disease_pred.py

## Run the Streamlit App

streamlit run disease_pred.py

## Open your browser

http://localhost:8501

## GOOD TO GO!!!

## Docker

## Build the Docker image:

docker build -t drug-predictor:latest .

## Run the Docker container:

docker run -p 8501:8501 drug-predictor:latest

## Open your browser

http://localhost:8501

## Features

- Fully offline: no internet connection required at runtime.
- Uses precomputed embeddings -> instant startup.
- Predicts Top 5 similar drugs with:
  - Drug name
  - Associated medical condition
  - Cosine similarity score
- Can run locally with Python or inside a Docker container.
- Built with Streamlit -> an interactive web interface.


