"""Класифікація Wine методом k найближчих сусідів."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "knn"


def add_color_ratio(frame):
    """Чотирнадцята ознака: відношення інтенсивності кольору до відтінку."""
    if (frame["hue"] <= 0).any():
        raise ValueError("Для обчислення індексу hue має бути додатним.")
    return frame.assign(color_hue_ratio=frame["color_intensity"] / frame["hue"])


def compare_features(X_train, y_train, folds, ratio, out_dir):
    """Окремий дослід із лабораторної № 2 при фіксованому k = 5."""
    variants = [
        ("13 raw", "passthrough", "passthrough"),
        ("13 scaled", "passthrough", StandardScaler()),
        ("14 scaled", ratio, StandardScaler()),
    ]
    rows = []
    for label, features, scaler in variants:
        pipeline = Pipeline([
            ("features", features),
            ("scale", scaler),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="f1_macro")
        row = {"variant": label, "mean_f1_macro": scores.mean()}
        for fold, score in enumerate(scores, start=1):
            row[f"fold_{fold}"] = score
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "features.csv", index=False)


def save_results(search, y_test, predicted, out_dir):
    """Записуємо всі варіанти пошуку, прогнози та тестові метрики."""
    cv = search.cv_results_
    trials = []
    for index, params in enumerate(cv["params"]):
        trial = {
            "features": 13 if params["features"] == "passthrough" else 14,
            "cv_macro_f1": cv["mean_test_score"][index],
            "selected": index == search.best_index_,
        }
        for name, value in params.items():
            if name.startswith("model__"):
                trial[name.removeprefix("model__")] = value
        for fold in range(search.n_splits_):
            trial[f"fold_{fold + 1}"] = cv[f"split{fold}_test_score"][index]
        trials.append(trial)
    pd.DataFrame(trials).to_csv(out_dir / "cv.csv", index=False)

    predictions = pd.DataFrame({"actual": y_test, "predicted": predicted})
    predictions.sort_index().to_csv(out_dir / "predictions.csv", index_label="sample_id")

    labels = search.classes_
    report = classification_report(y_test, predicted, labels=labels,
                                   output_dict=True, zero_division=0)
    accuracy = report.pop("accuracy")
    metrics = pd.DataFrame(report).T
    metrics["accuracy"] = accuracy
    metrics.to_csv(out_dir / "metrics.csv", index_label="group")

    matrix = confusion_matrix(y_test, predicted, labels=labels)
    confusion = pd.DataFrame(matrix, index=labels, columns=[f"pred_{n}" for n in labels])
    confusion.to_csv(out_dir / "confusion_matrix.csv", index_label="actual")

    correct = int((y_test == predicted).sum())
    print(f"Тест: {correct} / {len(y_test)} правильно")
    print(f"Accuracy: {accuracy:.4f} | macro-F1: {report['macro avg']['f1-score']:.4f}")
    print("Матриця помилок (рядок — справжній клас, стовпець — прогноз):")
    print(confusion.to_string())
    print(f"Результати: {out_dir}")


def main():
    # 1. Поділ.
    X, y = load_wine(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Wine: {len(X_train)} навчальних зразків, {len(X_test)} тестових.")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ratio = FunctionTransformer(add_color_ratio, validate=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Перевіряємо вплив масштабу та нової ознаки, використовуючи лише навчальні дані.
    compare_features(X_train, y_train, folds, ratio, RESULTS_DIR)

    # 3. Обираємо число сусідів і набір ознак. Кожен сусід має один голос.
    pipeline = Pipeline([
        ("features", "passthrough"),
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(metric="euclidean", weights="uniform")),
    ])
    parameters = {
        "features": ["passthrough", ratio],
        "model__n_neighbors": [3, 5, 7, 9, 11],
    }
    search = GridSearchCV(pipeline, parameters, cv=folds, scoring="f1_macro", refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    knn = best.named_steps["model"]
    print(f"kNN: k = {knn.n_neighbors}, ознак = {knn.n_features_in_}")
    print(f"Середня macro-F1 на валідації: {search.best_score_:.4f}")

    # 4–5. Разовий прогноз на тесті та оцінка якості.
    predicted = best.predict(X_test)
    save_results(search, y_test, predicted, RESULTS_DIR)

    # Демонстрація: знайти сусідів тестового зразка № 22 з початкового звіту.
    sample_id = 22
    if sample_id in X_test.index:
        sample = best[:-1].transform(X_test.loc[[sample_id]])
        distances, positions = knn.kneighbors(sample)
        neighbors = pd.DataFrame({
            "sample_id": sample_id,
            "neighbor_id": X_train.index[positions[0]],
            "neighbor_class": y_train.iloc[positions[0]].to_numpy(),
            "distance": distances[0],
        })
        neighbors.to_csv(RESULTS_DIR / "neighbors_22.csv", index=False)
        print("Зразок 22. Голоси сусідів:", neighbors["neighbor_class"].value_counts().to_dict())


if __name__ == "__main__":
    main()