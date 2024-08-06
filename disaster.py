import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import os
# import shutil
# classn="wildfire"
# folder="test"

# wild_source = f"data\\{folder}\\{classn}"
# wild_dest   = f"smalldata_save\\smalldata\\{folder}\\{classn}"
# wildfire    = os.listdir(f'data\\{folder}\\{classn}')

# for wild_testfile in wildfire[0:1000]:
#   shutil.move(os.path.join(wild_source,wild_testfile), wild_dest)



datadir = "smalldata"

img_height, img_width = 50,50 #250,250#350,350 # 100,100

train_ds = tf.keras.utils.image_dataset_from_directory(
  datadir,
  validation_split=0.2,
  subset="training",
  seed=128,
  image_size=(img_height, img_width),
  shuffle=True 
  )

val_ds = tf.keras.utils.image_dataset_from_directory(
  datadir,
  validation_split=0.2,
  subset="validation",
  seed=128,
  image_size=(img_height, img_width),
  shuffle=True 
  )

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = 2


model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
  ]),
  tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# model.summary()

# from tensorflow.keras.optimizers import Adam
# model.compile(optimizer=Adam(learning_rate=0.001)

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=32,
  epochs=100
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.show()