import unittest
import pandas as pd
from src.model_training import create_polynomial_features, initial_clustering, polynomial_regression, final_clustering, train_decision_tree, evaluate_model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/processed_data.csv')
        poly_data = create_polynomial_features(self.data)
        self.data, _ = initial_clustering(self.data)
        self.data, _ = polynomial_regression(self.data, poly_data)
        self.data, _ = final_clustering(self.data)
        self.X = self.data[['x', 'y', 'poly_regression']]
        self.y = self.data['cluster_2']

    def test_train_decision_tree(self):
        clf, X_test, y_test, y_pred = train_decision_tree(self.X, self.y)
        self.assertIsNotNone(clf)
        accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(precision, 0)
        self.assertGreaterEqual(recall, 0)
        self.assertGreaterEqual(f1, 0)

if __name__ == '__main__':
    unittest.main()
