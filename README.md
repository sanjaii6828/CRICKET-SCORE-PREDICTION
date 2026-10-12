# 🏏 Cricket Score Prediction Using Python and Machine Learning

## 📌 Project Description

Cricket Score Prediction is a machine learning project that predicts the final cricket score based on current runs, wickets, overs completed, and run rate. It supports T20 and ODI match formats and provides an interactive interface for score prediction and match analysis.

## 🚀 Features

* Predict final cricket scores using Machine Learning.
* Supports T20 and ODI formats.
* Uses pre-trained machine learning models.
* Generates sample cricket datasets.
* Provides an interactive web interface.
* Includes command-line score prediction.
* Stores model metadata and prediction logic.
* Includes automated tests.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Flask or Streamlit, depending on the implemented application
* HTML and CSS for the web interface

## 📂 Project Structure

* `.gitignore` — Excludes unnecessary files from Git.
* `__init__.py` — Initializes the Python package.
* `app.py` — Main application.
* `data_loader.py` — Loads cricket datasets.
* `generate_data.py` — Generates sample cricket data.
* `index.html` — Web interface template.
* `model_metadata.json` — Stores model information.
* `odi_matches.csv` — ODI training dataset.
* `odi_model.joblib` — Trained ODI prediction model.
* `predict_cli.py` — Command-line prediction tool.
* `predictor.py` — Prediction functions.
* `requirements.txt` — Required Python packages.
* `style.css` — Web interface styling.
* `t20_matches.csv` — T20 training dataset.
* `t20_model.joblib` — Trained T20 prediction model.
* `test_app.py` — Automated tests.
* `train.py` — Model training script.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd cricket_score_prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment on Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Train the models if required:

```bash
python train.py
```

Start the application:

```bash
python app.py
```

For command-line predictions:

```bash
python predict_cli.py
```

Run automated tests:

```bash
python -m pytest test_app.py
```

## 📊 Supported Formats

* **T20 Cricket:** 20 overs per innings.
* **ODI Cricket:** 50 overs per innings.

## 🎯 Objective

To demonstrate how machine learning can analyze cricket match situations and estimate final innings scores using historical match data.

## ⚠️ Disclaimer

Predictions are estimates and are not guaranteed actual scores. Prediction quality depends on the quality and quantity of training data.

## 👨‍💻 Author

Developed as a Python and Machine Learning project.
