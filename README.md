# marr-ml-projects

> Two graduate machine learning coursework projects: a tabular multiclass classifier comparison and a Convolutional Neural Network (CNN) controller for a simulated racing car.

## Overview

This repository holds two assignments completed for the Machine Learning course in the Robotics and Artificial Intelligence master's program at Sapienza University of Rome. The first project trains and compares six supervised classifiers on two synthetic 10-class datasets and profiles their accuracy against compute cost. The second project trains CNNs on labeled game frames to drive a car in the Gymnasium `CarRacing-v2` environment and evaluates them by both classification metrics and in-simulation reward. The work is academic: each project ships its source code, a written report, and result artifacts.

## Features

- **Six-classifier benchmark (Homework 1):** trains Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and XGBoost on two datasets (100 and 1000 features) with `scikit-learn` and `xgboost`.
- **Cross-validation comparison:** runs each model with and without 5-fold cross-validation and reports Accuracy, Precision, Recall, F1, and ROC/AUC.
- **Compute profiling:** measures wall-clock time and system resource usage (via `psutil`) alongside model quality, so model selection weighs accuracy against runtime.
- **Hyperparameter tuning:** uses `GridSearchCV` for per-model tuning and `StandardScaler` for feature standardization.
- **Two CNN architectures (Homework 2):** implements `model_cnn1` (stacked Conv/MaxPool blocks with dropout) and `model_cnn2` (5x5 filters with Batch Normalization) in Keras.
- **Class-imbalance handling:** loads images from per-class folders and optionally oversamples minority classes (`sklearn.utils.resample`) under a single `ENABLE_OVERSAMPLING` toggle, comparing balanced and unbalanced training.
- **Grid search over training settings:** sweeps batch size, learning rate, and epoch count, saving every trained model to disk as a Keras `.h5` file named by its hyperparameters.
- **Closed-loop evaluation:** runs saved models against `CarRacing-v2`, mapping each 96x96 frame to a discrete action and summing the episode's total reward.
- **Offline metrics:** scores saved models on a held-out test set with confusion matrices (rendered via `seaborn`) and macro-averaged classification metrics.

## Architecture

The two projects are independent and share no code; each is a small training-and-evaluation pipeline.

**Homework 1 — tabular classification**
1. `load_data.py` reads a CSV whose first column holds bracketed feature-vector strings, parses them into a NumPy feature matrix, and returns labels when present.
2. The notebook standardizes features with `StandardScaler` and splits data into training and evaluation sets.
3. Each of the six models trains both directly and under 5-fold cross-validation, with `GridSearchCV` tuning where time permits.
4. Quality metrics and resource usage are recorded per model; the final pick balances accuracy against cost (SVM for the 100-feature dataset, Random Forest for the 1000-feature dataset).

**Homework 2 — CNN driving controller**
1. `ML_Homework2_SourceCode_1793918.py` loads labeled 96x96 RGB frames from per-class subfolders, optionally oversamples to balance classes, and normalizes pixels to the 0–1 range.
2. It grid-searches batch size, learning rate, and epochs over both CNN architectures, compiling with the Adam optimizer and sparse categorical cross-entropy, and saves each result as an `.h5` model.
3. `ML_Homework2_evaluation_metrics_1793918.py` reloads the saved models and scores them on the test set, producing accuracy/precision/recall/F1 and confusion matrices.
4. `ML_Homework2_play_policy_template_1793918.py` loads a chosen model and runs it in the `CarRacing-v2` Gymnasium environment, predicting one discrete action per frame and reporting total episode reward and runtime.

## Tech Stack

| Category | Details |
| --- | --- |
| Language | Python (Jupyter Notebook for Homework 1) |
| ML / DL | scikit-learn, XGBoost, TensorFlow / Keras |
| RL environment | Gymnasium (`CarRacing-v2`, Box2D), pygame |
| Data / numerics | NumPy, pandas |
| Visualization | Matplotlib, seaborn |
| Profiling | psutil |

## Getting Started

Prerequisites: Python 3 with `scikit-learn`, `xgboost`, `tensorflow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, and `psutil`. Homework 2's simulation also needs `gymnasium[box2d]` and `pygame`.

The repository has no dependency manifest or central entry point; each file is run on its own. Note that the input datasets, training/test image folders, and trained `.h5` models are not committed, so the scripts run only after those assets are supplied locally.

```bash
# Homework 1: open and run the notebook
jupyter notebook "Homework 1/ML_Homework1_SourceCode_1793918.ipynb"

# Homework 2: train CNNs, then evaluate or simulate
python "Homework 2/ML_Homework2_SourceCode_1793918.py"
python "Homework 2/ML_Homework2_evaluation_metrics_1793918.py"
python "Homework 2/ML_Homework2_play_policy_template_1793918.py"
```

## Project Structure

```
Homework 1/
  ML_Homework1_SourceCode_1793918.ipynb   # six-classifier training and comparison
  load_data.py                            # CSV feature-vector parser
  ML_Homework1_Report_1793918.pdf         # written report
  ML_HMW1.pptx                            # slides
Homework 2/
  ML_Homework2_SourceCode_1793918.py            # CNN definitions + grid-search training
  ML_Homework2_evaluation_metrics_1793918.py    # offline metrics + confusion matrices
  ML_Homework2_play_policy_template_1793918.py  # CarRacing-v2 simulation runner
  ML_Homework2_Report_1793918.pdf               # written report
  ML_Homework_2_MER_1793918.xlsx                # metrics spreadsheet
  *.mp4                                          # simulation video
README.md
```

## Status

Academic coursework (completed assignments), not maintained as a product. No license file is present.
