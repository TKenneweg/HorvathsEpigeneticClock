import os
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys 
from mlp import MethylationDataset, AgePredictorMLP

from config import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from util import calibFunction, inverseCalibFunction
###############################################################################
# Configuration
###############################################################################

def main():
    dataset = MethylationDataset(SERIES_NAMES, DATA_FOLDER)

    train_indices = np.load("train_indices.npy", allow_pickle=True)
    test_indices = np.load("test_indices.npy", allow_pickle=True)
    X_data = dataset.X.numpy()
    y_data = dataset.y.numpy()
    X_train = X_data[train_indices]
    y_train = y_data[train_indices]
    X_test = X_data[test_indices]
    y_test = y_data[test_indices]



    # model = LinearRegression()
    model = ElasticNet(alpha=1e-3, l1_ratio=0.9) #linear regression with L1 and L2 regularization
    model.fit(X_train, calibFunction(y_train))
    preds = model.predict(X_test)



    test_error = np.mean(np.abs(inverseCalibFunction(preds) - y_test))
    test_median_error = np.median(np.abs(inverseCalibFunction(preds) - y_test))
    print(f"[RESULT] Test Error: {test_error:.2f}")
    print(f"[RESULT] Test Median Absolute Error: {test_median_error:.2f}")
    

    plt.figure()
    plt.scatter(y_test, inverseCalibFunction(preds), alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual vs Predicted Age")
    plt.tight_layout()
    plt.savefig("age_scatter_plot.png")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig('age_scatter_plot.png', dpi=300)
    plt.show()


    # Print model parameters
    print("\n#####################################\n")
    print("Model coefficients:", model.coef_)
    num_large_coeffs = np.sum(np.abs(model.coef_) > 0)
    print(f"Number of coefficients larger than 0: {num_large_coeffs}")
    plt.figure()
    plt.scatter(range(len(model.coef_)), model.coef_, alpha=0.5)
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title("Scatter Plot of Model Coefficients")
    plt.tight_layout()
    plt.savefig("coefficients_scatter_plot.png")
    plt.show()


if __name__ == "__main__":
    main()


    # # 7. Plot a histogram of errors (pred - actual)
    # errors = preds - y_test
    # plt.hist(errors, bins=20, edgecolor="black")
    # plt.axvline(x=0, color="red", linestyle="--")
    # plt.title("Histogram of Age Prediction Errors (Linear Regression)")
    # plt.xlabel("Error (Predicted - Actual)")
    # plt.ylabel("Number of Samples")
    # plt.tight_layout()
    # plt.show()
