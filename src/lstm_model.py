import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_features(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def train_lstm(data, look_back=1):
    X, y = create_lstm_features(data[['x']].values, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=1, verbose=2)
    return model, look_back

def predict_future_position_lstm(model, data, future_steps=5, look_back=1):
    predictions = []
    current_step = data[-look_back:]
    for _ in range(future_steps):
        prediction = model.predict(np.reshape(current_step, (1, look_back, 1)))
        predictions.append(prediction[0, 0])
        current_step = np.append(current_step[1:], prediction)
    return predictions

def train_decision_tree(data):
    X = data[['time', 'x', 'y']]
    y = data['cluster_1']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, X_test, y_test, y_pred

def evaluate_classification_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    real_data = pd.read_csv('data/real_data.csv')
    preprocessed_data = pd.read_csv('data/preprocessed_data.csv')

    train_data = preprocessed_data.iloc[:15]
    test_data = real_data.iloc[15:20]
    test_times = test_data['time'].values.reshape(-1, 1)

    model, look_back = train_lstm(train_data)

    future_positions = predict_future_position_lstm(model, test_data['x'].values, future_steps=len(test_times), look_back=look_back)

    actual_positions = test_data['x'].values
    mse = mean_squared_error(actual_positions, future_positions)
    r2 = r2_score(actual_positions, future_positions)
    
    print("Tahmin Edilen Gelecek Konumlar (X koordinatı):", future_positions)
    print("Gerçek Konumlar (X koordinatı):", actual_positions)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    kmeans = KMeans(n_clusters=3)
    preprocessed_data['cluster_1'] = kmeans.fit_predict(preprocessed_data[['x', 'y']])
    
    clf, X_test, y_test, y_pred = train_decision_tree(preprocessed_data)
    accuracy, precision, recall, f1 = evaluate_classification_model(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
