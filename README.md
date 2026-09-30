# Crop Suitability Classification Using Machine Learning

## Project Overview
This project develops a machine learning-based crop suitability classification system that predicts a suitable crop category from soil and environmental conditions.

## Problem Statement
Classify whether environmental and soil conditions are suitable for selected crop categories using measurable agricultural parameters.

## Objectives
- Analyze an agricultural dataset.
- Perform data cleaning and exploratory data analysis.
- Train foundational classification algorithms.
- Compare model performance.
- Select a final model using evaluation metrics.
- Build a Streamlit prediction application.

## Dataset
**Crop Recommendation Dataset**  
Source: Kaggle  
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Input features:
N, P, K, temperature, humidity, ph, rainfall

Target:
label

## Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

## Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Application
Run:
```bash
python -m streamlit run src/app.py
```

## Project Structure
```text
Crop-Suitability-Classification/
├── data/
├── notebooks/
├── src/
│   ├── eda.py
│   ├── train.py
│   └── app.py
├── models/
├── plots/
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations
This is an educational prototype. Real-world crop suitability may also depend on location, soil type, irrigation, season, weather history and expert agricultural assessment.

## Developer
Niharika Tiwari


