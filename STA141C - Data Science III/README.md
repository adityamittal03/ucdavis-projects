# STA141C - Data Science III

This project predicts flight delays using several classification models trained on `airlines_delay.csv`.
The original work was developed in `code.ipynb` and summarized in `report.pdf`.

## Scripts
Python scripts in the `scripts` directory reproduce the main analyses:

- `eda.py` – generates exploratory plots and saves them as PNG files.
- `log_reg_ascent.py` – logistic regression using gradient ascent.
- `log_reg_descent.py` – logistic regression using gradient descent.
- `naive_bayes.py` – custom Naive Bayes classifier.
- `knn.py` – k-nearest neighbors classifier.

Each script loads and preprocesses the data, trains the model using 10 different train/test splits and reports the average accuracy and runtime.

## Usage
Run a script with Python from this directory. For example:

```bash
python -m scripts.eda
python -m scripts.log_reg_ascent
python -m scripts.log_reg_descent
python -m scripts.naive_bayes
python -m scripts.knn
```

Plots and metrics will be printed or saved in the working directory.
