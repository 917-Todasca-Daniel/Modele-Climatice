import pandas as pd
import numpy as np
import tensorflow as tf
from keras.losses import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils.dataset import get_full_dataset
from utils.model import get_model

good_houses = ["000.021.184.023", "000.021.252.071", "000.021.217.122", "000.021.175.156", "000.046.195.015",
               "000.021.229.231", "000.022.005.033", "000.022.001.092", "000.022.003.102", "000.022.013.205",
               "000.021.219.099", "000.021.211.094", "000.021.226.174", "000.021.191.008", "000.021.193.234",
               "000.021.201.094", "000.021.230.247", "000.021.216.100", "000.021.160.006", "000.021.162.108",
               "000.046.192.109"]

seed = 10
np.random.seed(seed)


X_train, X_test, y_train, y_test = get_full_dataset([good_houses[0]], True)
print(X_train, X_test, y_train, y_test, sep='\n\n')


model = get_model()
hist = model.fit(X_train, y_train,
                 batch_size=100,
                 epochs=120,
                 verbose=2,
                 validation_data=(X_test, y_test)
                 )

i = 0
correct = 0
mean = 0
losses = []

predictions = model.predict(X_test)
for initial_value in y_test:
    if abs(initial_value - predictions[i]) < 50 and not (250 <= initial_value <= 350):
        correct += 1
    elif 250 <= initial_value <= 350 and abs(initial_value - predictions[i]) < 25:
        correct += 1
    i += 1
    losses.append(100 * abs((initial_value - predictions[i - 1]) / initial_value))

mean /= len(y_test)
print("Accuracy model is ", correct * 100 / i, "% with MAE of ",
      mean_absolute_error(y_test, predictions))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# from tensorflow.python.client import device_lib
# print("\nHERE", device_lib.list_local_devices())
# print("HERE \n")
