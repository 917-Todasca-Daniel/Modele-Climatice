from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.losses import mean_absolute_error
from keras.regularizers import l1, l2
from keras.optimizers import Adam

def get_model():
  model = Sequential()

  model.add(BatchNormalization())
  model.add(Dense(30, activation='relu', kernel_regularizer=l2(100),))
  model.add(BatchNormalization())
  model.add(Dense(20, activation='relu', kernel_regularizer=l2(50),))
  model.add(BatchNormalization())
  model.add(Dense(15, activation='relu', kernel_regularizer=l2(10)))
  model.add(BatchNormalization())
  model.add(Dense(10, activation='relu', kernel_regularizer=l2(1),))
  # model.add(BatchNormalization())
  model.add(Dense(1, activation='relu',))

  # compile the keras model
  model.compile(
    optimizer = Adam(learning_rate=1e-4),
    loss='mean_absolute_error',
    metrics=["mean_absolute_error"]
    )

  # model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=["mae"])
  # model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mae"])
  return model

# print(get_model().build().summary())
