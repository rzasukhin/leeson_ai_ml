"""Класифікація Wine за допомогою дерева рішень."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "tree"


def add_color_ratio(frame):
    """Чотирнадцята ознака: відношення інтенсивності кольору до відтінку."""
    if (frame["hue"] <= 0).any():
        raise ValueError("Для обчислення індексу hue має бути додатним.")
    return frame.assign(color_hue_ratio=frame["color_intensity"] / frame["hue"])


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

    # 2–3. Дерево порівнює ознаку з порогом у її вихідних одиницях.
    pipeline = Pipeline([
        ("features", "passthrough"),
        ("model", DecisionTreeClassifier(criterion="gini", random_state=SEED)),
    ])
    parameters = {
        "features": ["passthrough", ratio],
        "model__max_depth": [2, 3, 4, 5, None],
        "model__min_samples_leaf": [1, 3, 5],
    }
    search = GridSearchCV(pipeline, parameters, cv=folds, scoring="f1_macro", refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    tree = best.named_steps["model"]
    print(f"Дерево: глибина = {tree.get_depth()}, мінімум у листку = {tree.min_samples_leaf}, "
          f"ознак = {tree.n_features_in_}")
    print(f"Середня macro-F1 на валідації: {search.best_score_:.4f}")

    # 4–5. Разовий прогноз на тесті та оцінка якості.
    predicted = best.predict(X_test)
    save_results(search, y_test, predicted, RESULTS_DIR)

    # Правила дерева можна відкрити в PyCharm і простежити шлях до певного класу.
    rules = export_text(tree, feature_names=list(tree.feature_names_in_), decimals=2)
    (RESULTS_DIR / "tree_rules.txt").write_text(rules, encoding="utf-8")
    print(rules)


if __name__ == "__main__":
    main()