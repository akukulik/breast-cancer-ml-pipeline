from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics

if __name__ == "__main__":
    import sys
    import train_model
    import preprocess
    import load_data
    from sklearn.model_selection import train_test_split
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "etl/data.csv"

    df = load_data.load_data(path)
    X, y, _ = preprocess.preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model.train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики:", metrics)
