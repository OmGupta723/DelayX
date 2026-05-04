import pandas as pd

print("Loading data...")

cols = ['FL_DATE', 'AIRLINE', 'AIRLINE_CODE', 'ORIGIN', 'DEST',
        'DEP_DELAY', 'ARR_DELAY', 'CANCELLED',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']

df = pd.read_csv('data/flights_sample_3m.csv', usecols=cols)

print("Original shape:", df.shape)

# Step 1 - Fill NaN delay columns with 0
delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
              'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
              'DEP_DELAY', 'ARR_DELAY']
df[delay_cols] = df[delay_cols].fillna(0)

# Step 2 - Remove rows where AIRLINE or ORIGIN or DEST is missing
df = df.dropna(subset=['AIRLINE', 'ORIGIN', 'DEST'])

# Step 3 - Add a new column - 1 means delayed, 0 means not delayed
df['IS_DELAYED'] = (df['ARR_DELAY'] > 15).astype(int)

# Step 4 - Add month and day of week from date
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['MONTH'] = df['FL_DATE'].dt.month
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek

# Step 5 - Remove cancelled flights
df = df[df['CANCELLED'] == 0]


print("Cleaned shape:", df.shape)
print("\nDelayed flights:", df['IS_DELAYED'].sum())
print("On time flights:", (df['IS_DELAYED'] == 0).sum())
print("\nSample of clean data:")
print(df.head())

# Step 6 - Save clean data
df.to_csv('data/clean_flights.csv', index=False)
print("\nclean_flights.csv saved successfully!")

# ```

# Press `Ctrl + S` then in terminal run:
# ```
# python step2_cleaning.py