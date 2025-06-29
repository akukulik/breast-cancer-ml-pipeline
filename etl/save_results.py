import json

def save_metrics(metrics, path="results/metrics.json"):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Метрики сохранены в {path}")

if __name__ == "__main__":
    import sys
    import evaluate

    metrics_path = sys.argv[1] if len(sys.argv) > 1 else "results/metrics.json"

    # Здесь для теста можно просто вызвать evaluate из __main__
    # Но обычно метрики передаются как аргумент из внешнего кода
    # Для демонстрации:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Быстрый тестовый код (можно удалить)
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    model = LogisticRegression(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    save_metrics(metrics, metrics_path)
