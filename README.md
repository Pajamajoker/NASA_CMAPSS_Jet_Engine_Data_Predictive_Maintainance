# Predicting Remaining Useful Life (RUL) of Aircraft Engines using CMAPSS Dataset

This repository contains all the work related to a machine learning pipeline that predicts the Remaining Useful Life (RUL) of aircraft engines using the [CMAPSS dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository). It includes research experiments, final models, a project report, and a **simulated production deployment** of the best-performing model (SVR).

## Slides link: https://docs.google.com/presentation/d/18JEjDlT24bjkFT8RdOxX__Coai8zP3engZf0OW3rmfM/edit?usp=sharing

## 📁 Folder Structure
```bash
.
├── dataset/                    # Raw CMAPSS dataset files
├── final_research/            # Final consolidated notebook and script
│   ├── final_model.ipynb      # Clean, final Jupyter notebook
│   └── final_model.py         # Equivalent Python script of final code
├── research_notebooks/        # Exploratory and experimental notebooks
│   └── *.ipynb                # File names clearly describe the purpose
├── report/                    # Final report written in structured format
│   └── final_report.pdf
├── simulated_deployment/      # Simulated production environment for SVR model
│   ├── build/                 # Training and serialization of the SVR model
│   │   └── train.py
│   ├── deploy/                # Simulated pipeline and monitoring
│   │   ├── monitor.py         # Displays engine health based on RUL
│   │   └── pipeline.py        # Simulates data stream and RUL prediction
│   └── model/                 # Saved model files

```
## 📊 Project Overview

The goal of this project is to predict the Remaining Useful Life (RUL) of turbofan engines using multivariate time-series sensor data. The solution includes:

- Feature engineering using rolling means and trend indicators
- RUL capping to focus learning on critical degradation regions
- Training and evaluation of multiple models:
  - **SVR (Support Vector Regressor)**
  - Random Forest Regressor
  - XGBoost Regressor
- Simulated deployment environment to demonstrate streaming inference

## 🚀 Simulated Deployment Instructions

The `simulated_deployment/` folder demonstrates how the best-performing model (SVR) could be used in a streaming setting.

### Step 1: Train and Save Model

```bash
cd simulated_deployment/build
python train.py
```
This script loads the dataset, trains the SVR model, and saves it in the model/ directory.

### Step 2: Start the Monitoring System
Open Terminal 1:

```bash
cd simulated_deployment/deploy
python monitor.py
```
This script keeps watching the latest predictions and prints a summary of the engine's health based on current RUL.

### Step 3: Start the Prediction Pipeline
Open Terminal 2 in parallel:

```bash
cd simulated_deployment/deploy
python pipeline.py
```

This simulates sensor data coming in from 100 engines cycle-by-cycle and runs the SVR model to predict their RUL in real-time.

## 🧪 Notebooks for Research
All intermediate experiments (e.g., feature exploration, model comparisons, parameter tuning, etc.) are in the research_notebooks/ directory. File names are self-explanatory.

## 📄 Final Report
The full academic-style report summarizing the methodology, models, evaluation metrics, hyperparameter choices, and insights is available in the report/ folder.

## 🧠 Key Highlights
- Used domain knowledge to cap RUL at 200 cycles to focus on end-of-life prediction

- Chose rolling window of size 10 cycles after validation on smoothing trends

- SVR model (with RBF kernel) emerged as the most balanced in accuracy and generalization

- Built a simulated pipeline mimicking a real-time RUL monitoring system


Use the following command to install all requirements:

```bash
pip install -r requirements.txt
```

## 📬 Contact
- For questions or collaborations, feel free to reach out to [Your Name] at [Your Email].