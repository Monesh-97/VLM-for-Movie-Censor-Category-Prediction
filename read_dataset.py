import pandas as pd

try:
    df = pd.read_excel('Dataset/dataset.xlsx')
    print("Columns:", df.columns.tolist())
    print("\nData:")
    print(df.head(10))
except Exception as e:
    print(f"Error reading Excel file: {e}")
