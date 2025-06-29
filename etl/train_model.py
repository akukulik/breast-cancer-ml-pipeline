from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    import sys
    import preprocess
    import load_data
    from sklearn.model_selection import train_test_split

    path = sys.argv[1] if len(sys.argv) > 1 else "etl/data.csv"

    df = load_data.load_data(path)
    X, y, _ = preprocess.preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    print("Модель обучена.")