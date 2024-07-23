import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

def create_features(data, lag=1):
    features = pd.DataFrame()
    for i in range(1, lag + 1):
        features[f'lag_{i}'] = data['x'].shift(i)
    features['y'] = data['x']
    features['time'] = data['time']
    features['y_next'] = data['x'].shift(-1)
    features.dropna(inplace=True)
    return features

def train_random_forest(data, lag=1):
    features = create_features(data, lag)
    X = features.drop(columns=['y', 'y_next'])
    y = features['y_next']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, lag

def predict_future_position_random_forest(model, data, future_steps=5, lag=1):
    predictions = []
    current_input = data[['time', 'x']].values[-lag:].reshape(1, -1)
    for _ in range(future_steps):
        next_pred = model.predict(current_input)[0]
        predictions.append(next_pred)
        current_input = np.roll(current_input, -1)
        current_input[0, -1] = next_pred
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

    model, lag = train_random_forest(train_data)

    future_positions = predict_future_position_random_forest(model, train_data, future_steps=len(test_times), lag=lag)
    
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
