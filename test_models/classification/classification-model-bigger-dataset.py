import pandas as pd
import numpy as np

'''
Testing our model for classification using dataset with more records (15,000 synthetic records)
'''

pd.set_option('display.max_columns', None)


# === PREPARING DATA
data = pd.read_csv("../../Data/sleep_mobile_stress_dataset_15000.csv")
data = data.sample(frac=1).reset_index(drop=True)

data = data.drop('user_id', axis=1)
data = pd.get_dummies(data, columns=['gender', 'occupation'])
data = data.astype(float)

data = (data - data.mean()) / data.std()

data['is_depressed'] = np.where(
    (data['mental_fatigue_score'] < 0.4) |
    (data['stress_level'] > 0.7) |
    (data['sleep_quality_score'] < 4),
    1.0, 0.0)

real_data = data.copy()

# TARGET
y = data['is_depressed'].values.reshape(15000, 1)
# FEATURES
X = data.drop('is_depressed', axis=1).values

rows, cols = X.shape

# === NEURAL NETWORK
class Mentally_Unwell_Prediction:
    """
    input layers: (13-1) = 12 columns
    learning data: 15 0000 rows
    hidden layers: 128
    output layers: is depressed (0-1) - 1 layer
    """
    # Initialize the model
    def __init__(self, hidden_layers = 128):
        self.input = cols
        self.output = 1
        self.hidden_units = hidden_layers

        # ZMIENILAM TEN FRAGMENT BO MI KRZYCZALO ZE TO NIE INT
        half_units = self.hidden_units // 2

        # Weights
        self.w1 = np.random.randn(self.input, self.hidden_units) * np.sqrt(2. / self.input)
        self.w2 = np.random.randn(self.hidden_units, half_units) * np.sqrt(2. / self.hidden_units)
        self.w3 = np.random.randn(half_units, self.output) * np.sqrt(2. / half_units)

        # Velocity
        self.v1 = np.zeros_like(self.w1)
        self.v2 = np.zeros_like(self.w2)
        self.v3 = np.zeros_like(self.w3)

        # Biases - initialized to 0
        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((half_units, 1))
        self.b3 = np.zeros((self.output, 1))
        # DO TEGO MIEJSCA

    # Foward move from input layer through hidden layers, multiplying neuron by weight
    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1.T, X.T) + self.b1
        self.a2 = self.ReLU(self.z2)

        self.z3 = np.dot(self.w2.T, self.a2) + self.b2
        self.a3 = self.ReLU(self.z3)

        self.z4 = np.dot(self.w3.T, self.a3) + self.b3
        self.a4 = self.sigmoid(self.z4)

        return self.a4

    # Rectified Linear Unit
    def ReLU(self, Z): return np.maximum(Z, 0)
    def sigmoid(self, z): return 1 / (1 + np.exp(-z))

    # Binary Cross Entropy (Log Loss)
    def _loss(self, predict, y):
        m = y.shape[0]
        logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
        loss =- np.sum(logprobs) / m
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        rows = X.shape[0]

        # Output Layer (Sigmoid + BCE)
        dz4 = predict - y.T
        self.dw3 = (1 / rows) * np.dot(self.a3, dz4.T)
        self.db3 = (1 / rows) * np.sum(dz4, axis=1, keepdims=True)

        # Hidden Layer 2 (ReLU)
        dz3 = np.dot(self.w3, dz4) * self.ReLU_prime(self.z3)
        self.dw2 = (1 / rows) * np.dot(self.a2, dz3.T)
        self.db2 = (1 / rows) * np.sum(dz3, axis=1, keepdims=True)

        # Hidden Layer 1 (ReLU)
        dz2 = np.dot(self.w2, dz3) * self.ReLU_prime(self.z2)
        self.dw1 = (1 / rows) * np.dot(X.T, dz2.T)
        self.db1 = (1 / rows) * np.sum(dz2, axis=1, keepdims=True)

    def ReLU_prime(self, z): return (z>0).astype(float)
    def sigmoid_prime(self, z): return self.sigmoid(z) * (1 - self.sigmoid(z))

    def _update(self, learning_rate=0.01):
        beta = 0.9
        self.v1 = beta * self.v1 + (1-beta) * self.dw1
        self.w1 = self.w1 - learning_rate * self.v1
        self.b1 = self.b1 - learning_rate * self.db1

        self.v2 = beta * self.v2 + (1 - beta) * self.dw2
        self.w2 = self.w2 - learning_rate * self.v2
        self.b2 = self.b2 - learning_rate * self.db2

        self.v3 = beta * self.v3 + (1 - beta) * self.dw3
        self.w3 = self.w3 - learning_rate * self.v3
        self.b3 = self.b3 - learning_rate * self.db3

    def train(self, X, y, iteration=1000, learning_rate=0.001, batch_size=32):
        rows = X.shape[0]

        for i in range(iteration):
            self._backward_propagation(X, y)
            self._update(learning_rate)

            if i % 100 == 0:
                full_y_hat = self._forward_propagation(X)
                predictions = (full_y_hat.T > 0.5).astype(float)
                accuracy = np.mean(predictions == y)

                print(f"Iter {i} | Loss: {self._loss(full_y_hat, y):.4f} | Accuracy: {accuracy * 100:.2f}%")

            if i % 200 == 0:
                learning_rate *= 0.95

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        y_hat = [1 if i[0] >= 0.5 else 0 for i in y_hat.T]
        return np.array(y_hat)

    def score(self, predict, y):
        predict = predict.flatten()
        y = y.flatten()
        cnt = np.sum(predict == y)
        return (cnt / len(y)) * 100

def train(seperator=12000):
    X_train = X[:seperator]
    X_test = X[seperator:]

    y_train = y[:seperator]
    y_test = y[seperator:]

    clr = Mentally_Unwell_Prediction()

    clr.train(X_train, y_train)
    pre_y = clr.predict(X_test)
    score = clr.score(pre_y, y_test)

    print(f'=== SCORE: {score:.2f}%')

    def show_comparison(model, X_test, y_test):
        probs = model.predict(X_test)
        predictions = (probs.T > 0.5).astype(int)

        comparison = pd.DataFrame({
            'Actual Healthstatus': y_test.flatten(),
            'Predicted Health status': predictions.flatten()
        })

        print("\n=== ACTUAL VS PREDICTED ===")
        print(comparison.tail(10))
        acc = (comparison['Actual Healthstatus']==comparison['Predicted Health status']).mean()
        print(f"\nAverage Error: {acc*100:.1f}")

    show_comparison(clr, X_test, y_test)

    return clr
clr = train()

# TESTOWANIE MODELU
import io

csv_data = """user_id,age,gender,occupation,daily_screen_time_hours,phone_usage_before_sleep_minutes,sleep_duration_hours,sleep_quality_score,stress_level,caffeine_intake_cups,physical_activity_minutes,notifications_received_per_day,mental_fatigue_score
user_night_shift,42,Male,Nurse,8.5,15,5.2,4.1,8.5,5.0,20,65,7.8
user_freelancer,29,Female,Designer,11.2,50,6.5,5.8,6.0,3.0,15,110,6.2
user_corporate_exec,48,Male,Manager,9.0,25,5.8,5.0,9.2,6.0,45,210,8.5
user_athlete,24,Male,Coach,3.5,10,8.5,9.2,2.5,1.0,180,40,3.0
user_social_media,21,Female,Influencer,13.5,90,6.0,4.5,5.5,2.0,30,500,7.1
user_minimalist,33,Other,Gardener,2.1,5,8.0,8.8,1.8,0.0,120,12,2.0
user_exhausted_parent,31,Female,Stay-at-home,6.2,40,4.5,3.2,8.9,4.5,60,85,9.1
user_it_expert,37,Male,Developer,12.5,60,6.8,5.5,4.8,4.0,25,35,5.9"""

new_samples = pd.read_csv(io.StringIO(csv_data))


def predict_new_users(model, new_data, original_df):
    if 'user_id' in new_data.columns:
        new_data = new_data.drop('user_id', axis=1)

    new_data = pd.get_dummies(new_data)

    train_features = original_df.drop(columns=['is_depressed'])
    model_columns = train_features.columns

    new_data = new_data.reindex(columns=model_columns, fill_value=0)

    new_data_scaled = (new_data - train_features.mean()) / train_features.std()

    X_custom = new_data_scaled.values
    raw_probs = model._forward_propagation(X_custom)

    print("\n=== FINAL TEST RESULTS ===")
    for i, prob in enumerate(raw_probs.T):
        risk = prob[0]
        status = "Unwell" if risk >= 0.5 else "Healthy"
        print(f"Test {i + 1}: Health -> Diagnose: {status}")

predict_new_users(clr, new_samples, real_data)