import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

def regression_buildin_copm():
    # PREPARING DATA

    data = pd.read_csv("../digital_diet_mental_health.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop('user_id', axis=1)

    data = pd.get_dummies(data, columns=['gender', 'location_type'])

    data['stress_phone_interaction'] = data['stress_level'] * data['phone_usage_hours']
    data['total_digital_load'] = data['phone_usage_hours'] + data['laptop_usage_hours'] + data['gaming_hours']

    y = data['sleep_duration_hours'].values
    X = data.drop(columns='sleep_duration_hours').values

    # SPLITING INTO TRAIN AND TEST DATASETS
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.2)

    # SCALING
    sc_X = MinMaxScaler()
    X_trainscaled=sc_X.fit_transform(X_train)
    X_testscaled=sc_X.transform(X_test)

    # MODEL
    '''  sklearn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu',            
    solver='adam',                 
    learning_rate_init=0.001,     
    batch_size=32,                
    max_iter=1000,                
    momentum=0.9                  
    )'''

    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32), # Mniejsza sieć często lepiej generalizuje przy małych danych
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,    # Mniejszy LR dla stabilności
        max_iter=2000,               # Więcej czasu na zbieżność
        early_stopping=True,         # Zapobiega przeuczeniu i "psuciu" R2
        validation_fraction=0.1,
        random_state=1
    )

    # TRENING
    sklearn_model.fit(X_trainscaled, y_train)

    # PREDICTION
    y_pred=sklearn_model.predict(X_testscaled)
    print("Przykładowe przewidywania:", y_pred[:5])
    print("Przykładowe wartości realne:", y_test[:5])
    score = r2_score(y_test, y_pred)
    print(f"The Score with {score:.4f}")

    return score

regression_buildin_copm()