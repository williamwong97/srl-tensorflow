# Part 1 - Building the CNN
import tensorflow as tf

import constant


def create_cnn_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, 3, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, 3, 3, activation='relu', padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        tf.keras.layers.Flatten(name='layers_flatten'),
        tf.keras.layers.Dense(256, activation='relu', name='layers_dense'),
        tf.keras.layers.Dropout(0.5, name='layers_dropout'),
        tf.keras.layers.Dense(26, activation='softmax', name='layers_dense_2')
    ])


model = create_cnn_model()

# Compiling The CNN
model.compile(
    optimizer=tf.optimizers.legacy.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# Log file for model training history
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=constant.LOG_DIR, histogram_freq=1)

# Part 2 Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'mydata/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'mydata/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

model.fit(
    training_set,
    steps_per_epoch=800,
    epochs=78,
    validation_data=test_set,
    validation_steps=6500,
    callbacks=[tensorboard_callback]
)

# Saving the model
model.save('Trained_model.h5')

#
# print(model.history.keys())
# import matplotlib.pyplot as plt
#
# # summarize history for accuracy
# plt.plot(model.history['accuracy'])
# plt.plot(model.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
#
# plt.plot(model.history['loss'])
# plt.plot(model.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()
