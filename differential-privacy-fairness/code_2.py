# ----
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# Load the COMPAS dataset bundled with the repository.
data_path = os.path.join(os.path.dirname(__file__), 'compas-scores.csv')
df = pd.read_csv(data_path)

# pre-processing
df = df[df['race'].isin(['African-American', 'Caucasian'])]
df = df[df['decile_score'] != 5]
df['predicted'] = df['decile_score'].apply(lambda x: 1 if x >= 6 else 0)
df['true'] = df['two_year_recid']
df['score'] = df['decile_score']

groups = ['African-American', 'Caucasian']

# equalized odds confusion matrix
for i in groups:
    subset = df[df['race'] == i]
    tn, fp, fn, tp = confusion_matrix(subset['true'], subset['predicted']).ravel()
    print(f"\nConfusion matrix for {i}:")
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

# probabilities for each score - sufficiency metric
scores = sorted(df['score'].unique())
print("\nProbabilities by score and group:")
for i in scores:
    print(f"\nScore = {i}")
    for group in groups:
        subset = df[(df['race'] == group) & (df['score'] == i)]
        print(f"  {group}: P(Y=1)={subset['true'].mean()}, P(Y=0)={1 - subset['true'].mean()}")

