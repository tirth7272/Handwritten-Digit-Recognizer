import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# LOAD AND PREPROCESS DATA
# =========================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape dataset for CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# =========================
# DATA AUGMENTATION
# =========================

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.10,
    height_shift_range=0.10
)

datagen.fit(x_train)

# =========================
# MODEL DEFINITION
# =========================

model = Sequential()

# First Convolution Block
model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(28, 28, 1)
))
model.add(BatchNormalization())

model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second Convolution Block
model.add(Conv2D(
    64,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(BatchNormalization())

model.add(Conv2D(
    64,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(10, activation='softmax'))

# =========================
# MODEL COMPILATION
# =========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================

# Stops training if validation accuracy stops improving
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Saves the best model automatically
checkpoint = ModelCheckpoint(
    'mnist_trained.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# =========================
# MODEL TRAINING
# =========================

batch_size = 128
epochs = 30

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# =========================
# MODEL EVALUATION
# =========================

score = model.evaluate(x_test, y_test, verbose=0)

print('\nTest Loss:', score[0])
print('Test Accuracy:', score[1])

print("\nBest model saved as mnist_trained.h5")