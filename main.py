from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
# import resnext, resnet_v1, resnet_v2, mobilenets, inception_v3, inception_resnet_v2, densenet
import resnet_v2
import numpy as np
import os
import pandas as pd
import pickle 
import cv2
from keras.models import load_model

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Training parameters
batch_size = 32
epochs = 32
data_augmentation = False
subtract_pixel_mean = True  # Subtracting pixel mean improves accuracy
base_model = 'resnet20'
# Choose what attention_module to use: cbam_block / se_block / None
attention_module = 'cbam_block'
model_type = base_model if attention_module==None else base_model+'_'+attention_module

# Input image dimensions.
input_shape = 50,50,3


depth = 20 # For ResNet, specify the depth (e.g. ResNet50: depth=50)
# model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth, attention_module=attention_module)
model = resnet_v2.resnet_v2(input_shape=input_shape, depth=depth, attention_module=attention_module)   
# model = resnext.ResNext(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = mobilenets.MobileNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_v3.InceptionV3(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = densenet.DenseNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[
        'MeanSquaredError'
    ]
)
              
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = "/content/drive/MyDrive/Hassan-CV-Project/CBAM_Models/CBAM_Trainings/"
model_name = 'data_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             mode="min",
                             monitor='val_mean_squared_error',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

train_data = pd.read_csv("/content/drive/MyDrive/Hassan-CV-Project/Aff-Wild/train_Hassan.csv")
test_data = pd.read_csv("/content/drive/MyDrive/Hassan-CV-Project/Aff-Wild/test_Hassan.csv")

train = train_data[["Paths","Valence","Arousal"]].loc[25000:45000]
test = test_data[["Paths","Valence","Arousal"]].tail(5000)
print("test gen sy pehly")
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_dataframe(
  dataframe=train,
  x_col="Paths",
  subset="training",
  y_col=["Valence","Arousal"],
  batch_size=batch_size,
  seed=11,
  shuffle=True,
  class_mode="raw",
  target_size=(50,50))

valid_generator = datagen.flow_from_dataframe(
  dataframe=train,
  x_col="Paths",
  subset="validation",
  y_col=["Valence","Arousal"],
  batch_size=batch_size,
  seed=11,
  shuffle=True,
  class_mode="raw",
  target_size=(50,50))

test_generator = test_datagen.flow_from_dataframe(
    test,
    directory=None,
    x_col="Paths",
    # y_col="Valence", 
    y_col=["Valence","Arousal"],
    # weight_col=None,
    target_size=(50, 50),
    color_mode="rgb",
    # classes=None,
    class_mode="raw"
)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model = load_model('/content/drive/MyDrive/Hassan-CV-Project/CBAM_Models/CBAM_Trainings/data_resnet20_cbam_block_model.026.h5')

h = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=callbacks
)

with open('/content/drive/MyDrive/Hassan-CV-Project/CBAM_Models/CBAM_Trainings/history', 'wb') as file_pi:
        pickle.dump(h.history, file_pi)

score = model.evaluate(test_generator, batch_size=batch_size)

with open('score5000', 'wb') as file_pi:
  pickle.dump(score, file_pi)

print('Test Mean Squared Error = ', score[0])
print('Test Accuracy = ', score[1])

prediction = model.predict(test_generator)
with open('predictions5000', 'wb') as file_pi:
  pickle.dump(prediction, file_pi)


