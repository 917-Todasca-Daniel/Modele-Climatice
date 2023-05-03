import numpy as np
import numpy

import pandas as pd

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from constants import PATH_TO_DATASET, GOOD_HOUSES

import matplotlib.pyplot as plt

seed = 72
numpy.random.seed(seed)

verbose = 3
cache_size = 1000


def get_accuracy(target_y, predict_y):
    i = 0
    accuracy = 0
    for predict in predict_y:
        if abs(predict - target_y[i]) < 50 and not (250 <= target_y[i] <= 350):
            accuracy += 1
        elif 250 <= target_y[i] <= 350 and abs(target_y[i] - predict) < 25:
            accuracy += 1
        i += 1
    return accuracy / i * 100


def read_dataset(name):
    df = pd.read_csv(name)
    dataset = np.array(df)
    target = dataset[:, [5]]
    variables = dataset[:, [0, 1, 2, 3, 4, 6, 7, 8]]
    uids = list(dataset[:, 0])
    return target, variables, list(set(uids))


def poly_kernel():
    return make_pipeline(
        StandardScaler(),
        SVR(
            kernel='poly',
            degree=3,
            gamma='auto',
            coef0=1.0,
            C=1,
            cache_size=cache_size,
            verbose=verbose
        )
    )


def rbf_kernel():
    return make_pipeline(
        StandardScaler(),
        SVR(
            kernel='rbf',
            gamma='auto',
            C=100,
            cache_size=cache_size,
            verbose=verbose
        )
    )


def sigmoid_kernel():
    return make_pipeline(
        StandardScaler(),
        SVR(
            kernel='sigmoid',
            gamma='scale',
            coef0=1.0,
            C=1.0,
            cache_size=cache_size,
            verbose=verbose
        )
    )


uid = GOOD_HOUSES[0]
print(uid)
path_to_dataset = PATH_TO_DATASET + 'joined_homes/' + uid + ".csv"
y, X, _ = read_dataset(path_to_dataset)

X_train = []
y_train = []
X_test = []
y_test = []
X_cross = []
y_cross = []
for i in range(len(X)):
    row = X[i]
    if row[0] == 2019:
        X_train.append(row)
        y_train.append(y[i][0])
    elif row[0] == 2020:
        X_test.append(row)
        y_test.append(y[i][0])
    elif row[0] == 2021:
        X_cross.append(row)
        y_cross.append(y[i][0])

if __name__ == "__main__":
    model = sigmoid_kernel()
    history = model.fit(X_train, y_train)

    print()
    print("Train Accuracy:", get_accuracy(y_train, model.predict(X_train)))
    print("Test Accuracy:", get_accuracy(y_test, model.predict(X_test)))
    print("R2 score:", model.score(X_test, y_test))
    print("Test MAE:", mean_absolute_error(y_test, model.predict(X_test)))
    print("Train MAE:", mean_absolute_error(y_train, model.predict(X_train)))

    hand_check = 0
    for i in range(len(y_test)):
        hand_check += abs(y_test[i] - model.predict([X_test[i]])[0])
    hand_check /= len(y_test)
    print("Hand check test MAE:", hand_check)


    def diagonal_plot(prediction_values, target_values, train_target_values, train_prediction_values, name=""):
        plt.plot(target_values, prediction_values, 'or')
        plt.plot(train_target_values, train_prediction_values, 'ob')
        plt.axline((0, 0), (10, 10), color='red')
        plt.xlabel('target')
        plt.ylabel('prediction')
        plt.title(name)
        plt.show()


    def difference_plot(prediction_values, target_values, train_target_values, train_prediction_values, name=""):
        plt.plot([k for k in range(len(prediction_values))],
                 [abs(target_values[k] - prediction_values[k]) for k in range(len(prediction_values))],
                 'or')
        plt.plot([k for k in range(len(train_prediction_values))],
                 [abs(train_target_values[k] - train_prediction_values[k]) for k in
                  range(len(train_prediction_values))],
                 'ob')
        plt.ylabel("Loss = | predict - target | ")
        plt.xlabel('index')
        plt.title(name)
        plt.show()


    diagonal_plot(model.predict(X_test), y_test, model.predict(X_train), y_train, "diagonal graph")
    difference_plot(model.predict(X_test), y_test, model.predict(X_train), y_train, "difference graph")
