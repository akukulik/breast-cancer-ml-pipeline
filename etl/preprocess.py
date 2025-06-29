import pandas as pd

def preprocess(df):
    """
    Предобработка:
    - Удаление колонок 'id', если есть
    - Переименование колонок
    - Удаление пропусков в target
    - Преобразование diagnosis в бинарную метку
    - Заполнение пропусков в признаках медианой
    - Нормализация признаков (возвращаем X, y и scaler)
    """
    from sklearn.preprocessing import StandardScaler

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)

    df = df.dropna(subset=['diagnosis'])
    df = df.drop(columns=['unnamed:_32'])
    df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

    df = df.fillna(df.median())

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    import sys
    import load_data
    path = sys.argv[1] if len(sys.argv) > 1 else "etl/data.csv"
    df = load_data.load_data(path)
    X, y, _ = preprocess(df)
    print(f"Данные обработаны: {X.shape[0]} строк, {X.shape[1]} признаков.")