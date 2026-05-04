import pandas as pd

cols = ['FL_DATE', 'AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'DEST',
        'DEP_DELAY', 'ARR_DELAY', 'CANCELLED',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']

df = pd.read_csv('data/flights_sample_3m.csv', usecols=cols)

print("Data loaded successfully!")
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])
print("\nFirst 5 rows:")
print(df.head())

# ```

# Press `Ctrl + S` then run:
# ```
# python step1_load_data.py