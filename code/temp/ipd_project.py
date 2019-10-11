# Imports
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Load data
data = loadtxt(r"../../data/raw/pima-indians-diabetes.data.txt", delimiter=",")

# Split to input matrix and output label
X = data[:, 0:8]
y = data[:, 8]

# Define keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile keras model
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]
              )

# Fit model
# early_stopping_monitor = EarlyStopping(patience=1)
model.fit(X, y,
          epochs=150,
          batch_size=25,
          verbose=1,
          # callbacks= [early_stopping_monitor]
          )

# Evaluate model on the same data
# TODO: split data to train/test datasets
_, accuracy  = model.evaluate(X, y)
print(f"Accuracy: {accuracy}.")
