import pandas as pd
import logging

def load_data(path):
    """
    Загружает CSV-датасет, возвращает DataFrame.
    """
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "etl/data.csv"
    data = load_data(path)
    print(f"Загружено {len(data)} строк.")