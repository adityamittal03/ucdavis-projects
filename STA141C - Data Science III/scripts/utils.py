import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(sample=10000, standardize=False):
    """Load airlines_delay.csv and optionally standardize predictors.
    Returns X, y arrays and the sampled dataframe."""
    df = pd.read_csv('airlines_delay.csv')

    # label encode categorical columns
    df['NumAirline'] = df['Airline'].astype('category').cat.codes
    df['NumAirportFrom'] = df['AirportFrom'].astype('category').cat.codes
    df['NumAirportTo'] = df['AirportTo'].astype('category').cat.codes

    df = df.sample(n=sample, random_state=0)
    X = df[['Length', 'NumAirline', 'NumAirportFrom', 'NumAirportTo', 'DayOfWeek']]
    y = df['Class']

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values
    return X, y.values, df
