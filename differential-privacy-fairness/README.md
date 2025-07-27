# MAT180 - Fairness & Privacy in ML

This repository contains the final project for **MAT180 - Privacy and Fairness in Machine Learning**.


This folder explores both differential privacy and fairness using the COMPAS recidivism scores.

## Contents
- `code/differential-privacy.py` – Differentially private K-Means clustering. The script saves plots to `plots/` and falls back to a synthetic data set if `kmeansexample.asc` is missing.
- `code/fairness.py` – Equalized Odds and Sufficiency metrics computed from `compas-scores.csv`.
- `dp_fairness_report.pdf` – Final writeup from the course project.

## Results
`differential-privacy.py` shows how adding Laplace and Gaussian noise impacts the K-Means objective and clustering accuracy. `fairness.py` prints confusion matrices for African-American and Caucasian groups and shows recidivism probabilities for each COMPAS score to show fairness differences between groups.
