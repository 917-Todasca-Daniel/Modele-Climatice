from keras.models import Sequential
from keras.layers import Dense
# from keras.losses import mean_absolute_error
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.initializers import RandomNormal

def get_nn_model():
  model = Sequential()

  model.add(Dense(30, activation='relu', kernel_regularizer=l1_l2(10), kernel_initializer=RandomNormal(mean=0., stddev=0.05)))
  model.add(Dense(20, activation='relu', kernel_regularizer=l1_l2(25), kernel_initializer=RandomNormal(mean=0., stddev=0.05)))
  model.add(Dense(15, activation='relu', kernel_regularizer=l1_l2(50), kernel_initializer=RandomNormal(mean=0., stddev=0.05)))
  model.add(Dense(10, activation='relu', kernel_regularizer=l1_l2(75), kernel_initializer=RandomNormal(mean=0., stddev=0.05)))
  model.add(Dense(1, activation='relu', kernel_regularizer=l1_l2(100), kernel_initializer=RandomNormal(mean=0., stddev=0.05)))

  # compile the keras model
  model.compile(
    optimizer = Adam(learning_rate=5e-3),
    loss='mean_absolute_error',
    metrics=["mean_absolute_error"]
    )
  return model
