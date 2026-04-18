"""
=======================================================================
  PORÓWNANIE MODELI KLASYFIKACYJNYCH — Biblioteki sklearn
=======================================================================
  Cel: Porównanie skuteczności modeli uczenia maszynowego
       w zadaniu przewidywania zdrowia psychicznego.

  Modele:
    1. Regresja Logistyczna    (parametry: C, solver, penalty)
    2. k-Najbliższych Sąsiadów (parametry: n_neighbors, weights, metric)
    3. Las Losowy              (parametry: n_estimators, max_depth, min_samples_leaf)
    4. MLP Classifier (NN)    (parametry: hidden_layer_sizes, learning_rate_init)

  Wyniki zapisywane do CSV z uwzględnieniem zarówno zbioru
  treningowego jak i testowego.
=======================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.makedirs("../test_results/classification", exist_ok=True)


# ===================================================================
#  1. PRZYGOTOWANIE DANYCH
# ===================================================================

def prepare_data():
    data = pd.read_csv("../digital_diet_mental_health.csv")
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)

    data = data.drop('user_id', axis=1)
    data = pd.get_dummies(data, columns=['gender', 'location_type'])
    data = data.astype(float)
    data = (data - data.mean()) / data.std()

    data['is_depressed'] = np.where(
        (data['mental_health_score'] < 0.4) |
        (data['stress_level'] > 0.7) |
        (data['weekly_anxiety_score'] > 0.8),
        1.0, 0.0
    )

    y = data['is_depressed'].values
    X = data.drop('is_depressed', axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[Dane] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[Dane] Klasa 1 (niezdrowi): {y.mean()*100:.1f}%\n")

    return X_train, X_test, y_train, y_test


# ===================================================================
#  2. PORÓWNANIE MODELI Z DOMYŚLNYMI PARAMETRAMI
# ===================================================================

def compare_default_models(X_train, X_test, y_train, y_test):
    """
    Trenuje i porównuje 4 modele z domyślnymi (sensownymi) parametrami.
    Odpowiada parametrom użytym w autorskim modelu NN.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP Classifier (NN)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            learning_rate_init=0.005,
            batch_size=32,
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
    }

    results = []
    print("=" * 65)
    print("  MODELE Z DOMYŚLNYMI PARAMETRAMI")
    print("=" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        test_f1   = f1_score(y_test, model.predict(X_test))

        print(f"  {name:<30} | train: {train_acc*100:.2f}%  test: {test_acc*100:.2f}%  F1: {test_f1:.3f}")
        results.append({
            "model":          name,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy":  round(test_acc,  4),
            "f1_score":       round(test_f1, 4),
            "param_name":     "default",
            "param_value":    "default"
        })

    return results


# ===================================================================
#  3. ANALIZA PARAMETRÓW — REGRESJA LOGISTYCZNA
# ===================================================================

def test_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
      • C (regularyzacja): [0.01, 0.1, 1, 10, 100]
      • solver: ['lbfgs', 'liblinear', 'saga', 'newton-cg']
      • penalty: ['l1', 'l2', 'elasticnet', None]
    """
    results = []

    print("\n[Logistic Regression] Testowanie parametru C (regularyzacja)...")
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "Logistic Regression",
            "param_name":     "C",
            "param_value":    str(C),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[Logistic Regression] Testowanie parametru solver...")
    for solver in ['lbfgs', 'liblinear', 'saga', 'newton-cg']:
        try:
            model = LogisticRegression(solver=solver, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            results.append({
                "model":          "Logistic Regression",
                "param_name":     "solver",
                "param_value":    solver,
                "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
                "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
                "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
            })
        except Exception as e:
            print(f"  Pominięto solver={solver}: {e}")

    print("[Logistic Regression] Testowanie parametru penalty...")
    for penalty, solver in [('l1','liblinear'), ('l2','lbfgs'), ('elasticnet','saga'), (None,'lbfgs')]:
        try:
            kwargs = {"l1_ratio": 0.5} if penalty == 'elasticnet' else {}
            model = LogisticRegression(penalty=penalty, solver=solver, max_iter=1000, random_state=42, **kwargs)
            model.fit(X_train, y_train)
            results.append({
                "model":          "Logistic Regression",
                "param_name":     "penalty",
                "param_value":    str(penalty),
                "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
                "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
                "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
            })
        except Exception as e:
            print(f"  Pominięto penalty={penalty}: {e}")

    return results


# ===================================================================
#  4. ANALIZA PARAMETRÓW — K-NEAREST NEIGHBORS
# ===================================================================

def test_knn(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
      • n_neighbors: [1, 3, 5, 10, 20]
      • weights: ['uniform', 'distance']
      • metric: ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    """
    results = []

    print("\n[KNN] Testowanie parametru n_neighbors...")
    for k in [1, 3, 5, 10, 20]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        results.append({
            "model":          "K-Nearest Neighbors",
            "param_name":     "n_neighbors",
            "param_value":    str(k),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[KNN] Testowanie parametru weights...")
    for w in ['uniform', 'distance']:
        model = KNeighborsClassifier(n_neighbors=5, weights=w)
        model.fit(X_train, y_train)
        results.append({
            "model":          "K-Nearest Neighbors",
            "param_name":     "weights",
            "param_value":    w,
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[KNN] Testowanie parametru metric...")
    for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
        model = KNeighborsClassifier(n_neighbors=5, metric=metric)
        model.fit(X_train, y_train)
        results.append({
            "model":          "K-Nearest Neighbors",
            "param_name":     "metric",
            "param_value":    metric,
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    return results


# ===================================================================
#  5. ANALIZA PARAMETRÓW — RANDOM FOREST
# ===================================================================

def test_random_forest(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
      • n_estimators: [10, 50, 100, 200]
      • max_depth: [None, 3, 5, 10]
      • min_samples_leaf: [1, 2, 5, 10]
    """
    results = []

    print("\n[Random Forest] Testowanie parametru n_estimators...")
    for n in [10, 50, 100, 200]:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "Random Forest",
            "param_name":     "n_estimators",
            "param_value":    str(n),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[Random Forest] Testowanie parametru max_depth...")
    for d in [None, 3, 5, 10]:
        model = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "Random Forest",
            "param_name":     "max_depth",
            "param_value":    str(d),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[Random Forest] Testowanie parametru min_samples_leaf...")
    for msl in [1, 2, 5, 10]:
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=msl, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "Random Forest",
            "param_name":     "min_samples_leaf",
            "param_value":    str(msl),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    return results


# ===================================================================
#  6. ANALIZA PARAMETRÓW — MLP CLASSIFIER
# ===================================================================

def test_mlp(X_train, X_test, y_train, y_test):
    """
    Testowane parametry:
      • hidden_layer_sizes: [(64,), (128,64), (128,64,32), (256,128,64)]
      • learning_rate_init: [0.01, 0.005, 0.001, 0.0005]
      • activation: ['relu', 'tanh', 'logistic']
      • batch_size: [16, 32, 64, 128]
    """
    results = []

    print("\n[MLP] Testowanie parametru hidden_layer_sizes...")
    for hl in [(64,), (128, 64), (128, 64, 32), (256, 128, 64)]:
        model = MLPClassifier(hidden_layer_sizes=hl, max_iter=1000,
                              early_stopping=True, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "MLP Classifier (NN)",
            "param_name":     "hidden_layer_sizes",
            "param_value":    str(hl),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[MLP] Testowanie parametru learning_rate_init...")
    for lr in [0.01, 0.005, 0.001, 0.0005]:
        model = MLPClassifier(learning_rate_init=lr, max_iter=1000,
                              early_stopping=True, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "MLP Classifier (NN)",
            "param_name":     "learning_rate_init",
            "param_value":    str(lr),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[MLP] Testowanie parametru activation...")
    for act in ['relu', 'tanh', 'logistic', 'identity']:
        model = MLPClassifier(activation=act, max_iter=1000,
                              early_stopping=True, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "MLP Classifier (NN)",
            "param_name":     "activation",
            "param_value":    act,
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    print("[MLP] Testowanie parametru batch_size...")
    for bs in [16, 32, 64, 128]:
        model = MLPClassifier(batch_size=bs, max_iter=1000,
                              early_stopping=True, random_state=42)
        model.fit(X_train, y_train)
        results.append({
            "model":          "MLP Classifier (NN)",
            "param_name":     "batch_size",
            "param_value":    str(bs),
            "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
            "test_accuracy":  round(accuracy_score(y_test,  model.predict(X_test)),  4),
            "f1_score":       round(f1_score(y_test, model.predict(X_test)), 4),
        })

    return results


# ===================================================================
#  7. URUCHOMIENIE I ZAPIS WYNIKÓW
# ===================================================================

def main():
    X_train, X_test, y_train, y_test = prepare_data()

    # Zbieramy wszystkie wyniki
    all_results = []
    all_results += compare_default_models(X_train, X_test, y_train, y_test)
    all_results += test_logistic_regression(X_train, X_test, y_train, y_test)
    all_results += test_knn(X_train, X_test, y_train, y_test)
    all_results += test_random_forest(X_train, X_test, y_train, y_test)
    all_results += test_mlp(X_train, X_test, y_train, y_test)

    df = pd.DataFrame(all_results)

    # Zapis do CSV
    out_path = "../test_results/classification/classification_builtin_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWszystkie wyniki zapisane do: {out_path}")

    # Podsumowanie — najlepsza konfiguracja per model
    print("\n" + "=" * 65)
    print("  NAJLEPSZA KONFIGURACJA DLA KAŻDEGO MODELU")
    print("=" * 65)
    best = df.loc[df.groupby('model')['test_accuracy'].idxmax()]
    print(best[['model', 'param_name', 'param_value', 'train_accuracy', 'test_accuracy', 'f1_score']].to_string(index=False))

    return df


if __name__ == "__main__":
    main()