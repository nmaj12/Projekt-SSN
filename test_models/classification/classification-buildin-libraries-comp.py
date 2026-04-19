import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score




def classification_buildin_models_comparison():
    # PREPARING DATA

    data = pd.read_csv("../../Data/digital_diet_mental_health.csv")

    data = data.drop('user_id', axis=1)
    data = pd.get_dummies(data, columns=['gender', 'location_type'])
    data = data.astype(float)

    data = (data - data.mean()) / data.std()

    data['is_depressed'] = np.where(
        (data['mental_health_score'] < 0.4) |
        (data['stress_level'] > 0.7) |
        (data['weekly_anxiety_score'] > 0.8),
        1.0, 0.0)

    y = data['is_depressed'].values
    X = data.drop('is_depressed', axis=1).values

    # SPLITING INTO TRAIN AND TEST DATASETS
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.2)

    # SCALING
    sc_X = MinMaxScaler()
    X_trainscaled=sc_X.fit_transform(X_train)
    X_testscaled=sc_X.transform(X_test)
    
    # MODELS
    models = {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1),
            "MLP Classifier (NN)": MLPClassifier(
                hidden_layer_sizes=(128, 64), 
                learning_rate_init=0.005,
                batch_size=32,
                activation='relu', 
                solver='adam', 
                max_iter=2000, 
                early_stopping=True, 
                random_state=1
            )
        }
    
    # TRENING
    results_list = []

    print("\n--- TRAINING MODELS ---\n")
    for name, model in models.items():
        model.fit(X_trainscaled, y_train)
        preds = model.predict(X_testscaled)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        results_list.append({
            "Model": name, 
            "Accuracy": acc, 
            "F1-Score": f1
        })

    # ADDING MY NN 
        my_results = pd.read_csv("../../test_results/classification/default_classification_results.csv")
        my_acc = accuracy_score(my_results['Actual_Status'], my_results['Predicted_Status'])
        my_f1 = f1_score(my_results['Actual_Status'], my_results['Predicted_Status'])


    results_list.append({
        "Model": "My Custom NN", 
            "Accuracy": my_acc, 
            "F1-Score": my_f1
    })

    # 5. BEST RESULTS
    df_results = pd.DataFrame(results_list)
    # SORTING BY R2
    df_results = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    print("BEST MODELS:")
    print("=" * 60)
    print(df_results.to_string())
    print("=" * 60)
    
    best_model = df_results.iloc[0]['Model']
    print(f"\nTHE BEST ONE: {best_model}")
    
    return df_results

classification_buildin_models_comparison()