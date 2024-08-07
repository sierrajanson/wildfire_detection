##############################################
# Wildfire Detection from Satellite Images --#
##############################################
"""
Author: Sierra Janson
Created: August 2024
About: This program will process 16000+ satellite images and train a model to determine if the image contains a wildfire
Note: This project is a work in progress, and more finetuning needs to be done to prevent the model from overfitting 
      as (96%+ training data accuracy was achieved, but only 78% validation accuracy has been achieved)
"""



# importing modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


##################################################
# Retrieving images from local image directory --#
##################################################
# images are normalized to values between 0 and 1, and resized to be (50,50)
# the images have already been checked for corruption by a different script (as unfortunately there were a couple corrupt files included in the dataset)
# another opportunity for processing is experimenting with converting the images to grayscale and seeing how that affect's the model's output

# initializing constants
DATADIR               = "smalldata"
IMG_HEGIHT, IMG_WIDTH = 50,50 
DATA_SEED             = 124
VALIDATION_SPLIT      = 0.2
OUTPUT_CLASSES        = 2
DATA_SHUFFLED         = True


# utilizing tf.keras to create train & val ds
train_ds = tf.keras.utils.image_dataset_from_directory(
  DATADIR,
  validation_split=VALIDATION_SPLIT,
  subset="training",
  seed=DATA_SEED,
  image_size=(IMG_HEGIHT, IMG_WIDTH),
  shuffle=DATA_SHUFFLED 
  )

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATADIR,
  validation_split=0.2,
  subset="validation",
  seed=DATA_SEED,
  image_size=(IMG_HEGIHT, IMG_WIDTH),
  shuffle=DATA_SHUFFLED 
  )


# initializing cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#########################
# Training Model -------#
#########################

# mitigating overfitting 
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


# initializing some hyperparameters 
DROPOUT_RATE         = 0.2
ACTIVATION_FUNC      = 'relu'
OPTIMIZER            = 'adam'
EPOCHS               = 100
BATCH_SIZE           = 32


# CNN model architecture!
# multiple convolutional layers and max pooling layers are provided to give the model an opportunity to deeply learn image features
# data augmentation techniques & a drop out rate is implemented as this model struggled with overfitting (achieving 96%+ training data accuracy but 76% validation accuracy)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),           # normalize images so computer resources are used more efficiently!
  data_augmentation,
  tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=ACTIVATION_FUNC),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=ACTIVATION_FUNC),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=ACTIVATION_FUNC),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(DROPOUT_RATE),
  tf.keras.layers.Dense(128, activation=ACTIVATION_FUNC),
  tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax')
])


model.compile(
  optimizer=OPTIMIZER,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# commence training!
run = model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS
)

##################################
# Graphing accuracy of model ----#
##################################

plt.plot(run.history['accuracy'])
plt.title('Accuracy of Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc = 'lower right')
plt.show()


#####################
# Testing model ----#
#####################

# open unseen images & have model predict class 