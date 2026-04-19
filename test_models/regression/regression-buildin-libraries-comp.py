import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

def regression_buildin_models_comparison():
    # PREPARING DATA

    data = pd.read_csv("../../Data/digital_diet_mental_health.csv")
    data = data.drop('user_id', axis=1)

    data = pd.get_dummies(data, columns=['gender', 'location_type'])
    data = data.reindex(sorted(data.columns), axis=1)

    data['stress_phone_interaction'] = data['stress_level'] * data['phone_usage_hours']
    data['total_digital_load'] = data['phone_usage_hours'] + data['laptop_usage_hours'] + data['gaming_hours']

    y = data['sleep_duration_hours'].values / 10.0
    X = data.drop(columns='sleep_duration_hours').values

    # SPLITING INTO TRAIN AND TEST DATASETS
    X_train, X_test = X[:1600], X[1600:]
    y_train, y_test = y[:1600], y[1600:]

    # SCALING
    sc_X = MinMaxScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    # MODELS
    models = {
            "Linear Regression": LinearRegression(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=1),
            "MLP Regressor (NN)": MLPRegressor(
                hidden_layer_sizes=(128, 64), 
                learning_rate_init=0.005,
                batch_size=32,
                activation='relu', 
                solver='adam', 
                max_iter=2000, 
                momentum=0.9,
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
        
        mae = mean_absolute_error(y_test, preds)*10
        
        results_list.append({
            "Model": name, 
            "MAE (h)": mae
        })

    # ADDING MY NN
    my_custom_results = pd.read_csv("../../test_results/regression/default_regression_results.csv")
    my_mae = mean_absolute_error(my_custom_results['Actual_Hours'], my_custom_results['Predicted_Hours'])

    results_list.append({
            "Model": "My Custom NN", 
            "MAE (h)": my_mae
        })

    # 5. BEST RESULTS
    df_results = pd.DataFrame(results_list)
    # SORTING BY R2
    df_results = df_results.sort_values(by="MAE (h)", ascending=True).reset_index(drop=True)

    print("BEST MODELS:")
    print("=" * 60)
    print(df_results.to_string())
    print("=" * 60)
    
    best_model = df_results.iloc[0]['Model']
    print(f"\nTHE BEST ONE: {best_model}")
    
    return df_results

regression_buildin_models_comparison()