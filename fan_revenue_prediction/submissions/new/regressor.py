from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = LinearRegression()

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)