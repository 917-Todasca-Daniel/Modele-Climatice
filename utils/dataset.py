import pandas as pd
import numpy as np
import random

seed = 10
np.random.seed(seed)

def read_dataset(name, log=True):
    name = "./Radon dataset/joined_homes" + ('_log' if log else '') + '/' + name + ".csv"
    df = pd.read_csv(name)
    dataset = np.array(df)
    # np.random.shuffle(dataset)
    # 5 - val1h
    target = dataset[:, [5]]
    # 0 - year, 1 - month, 2 - day, 3 - ord, 4 - hour, 6 - temp, 7 - hum, 8 - val CO2
    variables = dataset[:, [3, 4, 6, 7, 8]]
    year = dataset[:, [0]]
    # print('target', target, "variables", variables, "year", year, sep='\n\n')
    return variables, target, year

def split_dataset(X, y, year):
    random.seed()
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(X)):
        row = X[i]
        # print(row)
        # if random.randrange(0, 10) < 8:
        if year[i][0] == 2019:
        # if row[0] < 300:
            X_train.append(row)
            y_train.append(y[i][0])
        else:
        # elif year[i][0] == 2020:
            X_test.append(row)
            y_test.append(y[i][0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


def get_houses_split_dataset(house_names, log=True):
    X_train, X_test, y_train, y_test = [], [], [], [] # np.array([]), np.array([]), np.array([]), np.array([])
    for house_name in house_names:
        X_train_curr, X_test_curr, y_train_curr, y_test_curr = \
            split_dataset(*read_dataset(house_name, log))
        # print(X_train_curr.shape, X_test_curr.shape)
        if X_train == []:
            X_train = X_train_curr
        else:
            np.concatenate((X_train, X_train_curr), axis=0)
        if X_test == []:
            X_test = X_test_curr
        else:
            np.concatenate((X_test, X_test_curr), axis=0)
        if y_train == []:
            y_train = y_train_curr
        else:
            np.concatenate((y_train, y_train_curr), axis=0)
        if y_test == []:
            y_test = y_test_curr
        else:
            np.concatenate((y_test, y_test_curr), axis=0)
    return X_train, X_test, y_train, y_test

def get_house_split_dataset(house_name, log=True):
    return get_houses_split_dataset([house_name], log)

