# Carbon-Paradox-Analysis
A data-driven exploration of the "Carbon Paradox" using CRISP-DM methodology. Features K-Means clustering, SHAP game-theoretic attribution, and PyTorch LSTM forecasting deployed via an interactive Streamlit dashboard.

## Cloud Deployment
You may visit web version of our project via https://carbon-paradox-analysis-yjldj6m7krkxuu3b5ess3y.streamlit.app/

## Acknowledgment
This project is a Group Project from course of "Programming for Data Science" at ISEG, finished by 

*BIN XU*, (Github Account Holder, Coding and Cloud Deployment)

*XILUN LI*, *TSE GA WING*, *AZIZBEK MUMINOV*, (Writing Parts, Report and Presentation)

## Project Overview
This project investigates the "Carbon Paradox" using Data Science and Machine Learning techniques (K-Means, Random Forest, SHAP, and LSTM via PyTorch). The entire workflow strictly follows the **CRISP-DM** methodology, from Data Understanding to Deployment (Actionable Knowledge).

** Data Source Declaration:** All datasets were officially harvested via direct downloads from **Our World in Data (OWID)** and the **World Bank**. **NO Kaggle datasets were used** in this project, strictly adhering to the course instructions.

---

## How to Run the Application
We have developed a fully functional, interactive Python Web Application using `Streamlit` to present our multidimensional analysis and 2030 AI forecasts.

### Step 1: Install Dependencies
Ensure you have Python installed. Open your terminal or command prompt in this project folder and run:
`pip install -r requirements.txt`

### Step 2: Launch the Dashboard
Run the following command to start the web application:
`streamlit run app.py`

*(The interactive dashboard will automatically open in your default web browser, typically at `http://localhost:8501`).*

---

## Directory Structure
* `app.py`: The main interactive Python Web Application containing the CRISP-DM dashboard.
* `main.ipynb`: The fully-functional Jupyter Notebook containing complete experimental logs, data cleaning logic, and PyTorch deep learning training loops.
* `dataset/`: The directory containing our raw and merged official CSV datasets.
* `requirements.txt`: The Python package dependencies required to run the code.
