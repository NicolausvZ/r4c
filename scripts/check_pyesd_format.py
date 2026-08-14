import pandas as pd

df = pd.read_csv('examples/pyESD/data/predictors_train_1958-2000.csv', nrows=5)
print(f"Columns: {list(df.columns)}")
print(df.head())