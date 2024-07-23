import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def create_polynomial_features(data, degree=2):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(data[['time']]), poly

def train_polynomial_regression(data, degree=2):
    X_poly, poly = create_polynomial_features(data, degree)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, data['x'])
    return lin_reg, poly

def predict_future_position(model, poly, future_times):
    X_poly_future = poly.transform(future_times.reshape(-1, 1))
    return model.predict(X_poly_future)

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

    model, poly = train_polynomial_regression(train_data)

    future_positions = predict_future_position(model, poly, test_times)
    
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
