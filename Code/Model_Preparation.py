import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tqdm import tqdm
import logging
import math
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import tensorflow_model_optimization as tfmot
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from keras_cv_attention_models import efficientvit_b, efficientvit_m, efficientformer, fastvit, fastervit, mobilevit, tinyvit, repvit
from tensorflow.keras.applications import VGG16, InceptionV3, Xception, EfficientNetB7, VGG19, DenseNet121, MobileNetV3Large, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetV2S


print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# --------------------------------------------- #

def F1_Score(y_true, y_pred): # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    
# --------------------------------------------- #

train_dir  = "/path/to/training/set"
val_dir = "/path/to/validation/set
test_dir = "/path/to/testing/set"
 
batch_size = 32
img_size = 224

# --------------------------------------------- #

datagen = ImageDataGenerator(
    rescale = 1./255 
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size
)

valid_data = datagen.flow_from_directory(
    val_dir, 
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle = False
)

test_data = datagen.flow_from_directory(
    test_dir, 
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle = False
)

# --------------------------------------------- #

train_steps_per_epoch = math.ceil(train_data.samples/batch_size)
valid_steps_per_epoch = math.ceil(valid_data.samples/batch_size)
test_steps_per_epoch = math.ceil(test_data.samples/batch_size)

print(train_steps_per_epoch)
print(valid_steps_per_epoch)
print(test_steps_per_epoch)

tf.keras.backend.clear_session()

num_classes = train_data.num_classes
print(f"Number of classes: {num_classes}")

# --------------------------------------------- #
# --------------------------------------------- #
# ---------------MODEL BUILDING---------------- #
# --------------------------------------------- #
# --------------------------------------------- #


base_model = repvit.RepViT_M11(input_shape = (224, 224, 3), num_classes = 0) ## Can be modified to include any of the models from keras_cv_attention_models
top = GlobalAveragePooling2D()(base_model.output)
#top = base_model.output
mid = Dense(1024, activation = 'relu')(top)
drop = Dropout(0.3)(mid)
out = Dense(num_classes, name = 'output', activation = 'softmax')(drop)

model = keras.Model(inputs = base_model.input, outputs= out)

for layer in base_model.layers:
        layer.trainable = False

optimizer = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy', F1_Score])

model.summary()

# --------------------------------------------- #

num_epochs = 15
K.clear_session()
history = model.fit(train_data, steps_per_epoch=train_steps_per_epoch,
                    validation_data=valid_data, validation_steps=valid_steps_per_epoch,
                    epochs=num_epochs)
                    
# --------------------------------------------- #

print("Checking on validation data: ")

predicted = np.argmax(model.predict(x=valid_data, steps=valid_steps_per_epoch),axis=1)

actual = []
for i in range(0,int(valid_steps_per_epoch)):
    actual.extend(np.array(valid_data[i][1])) 

actual=np.argmax(np.array(actual),axis=1)


cr=classification_report(actual,predicted, output_dict = False)
cm=confusion_matrix(actual,predicted)   

print("Classification report: \n", cr)  
print("Classification matrix: \n", cm)  

print("-"*100)

# --------------------------------------------- #

print("Checking on testing data: ")
predicted = np.argmax(model.predict(x=test_data, steps=test_steps_per_epoch),axis=1)

actual = []
for i in range(0,int(valid_steps_per_epoch)):
    actual.extend(np.array(valid_data[i][1])) 

actual=np.argmax(np.array(actual),axis=1)


cr=classification_report(actual,predicted, output_dict = False)
cm=confusion_matrix(actual,predicted)   

print("Classification report: \n", cr)  
print("Classification matrix: \n", cm)  

print("-"*100)

# --------------------------------------------- #

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('/path_to_save_directory/(F32)RepViT_M11.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("tflite_model successfully saved (float32 default model).")
    
print("-"*100)

# --------------------------------------------- #

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()
with open('/path_to_save_directory/(F16)RepViT_M11.tflite', 'wb') as f:
    f.write(tflite_fp16_model)


print("tflite_model successfully saved (float16 quantized model).")

print("-"*100)

# --------------------------------------------- #

def representative_data_gen():
    for i in range(100):
        image, _ = train_data.next()
        yield [image.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_int8_model = converter.convert()
with open('/path_to_save_directory/(I8)RepViT_M11.tflite', 'wb') as f:
    f.write(tflite_int8_model)

print("tflite_model successfully saved (int8 quantized model).")
print("-"*100)

# --------------------------------------------- #

print("Performing QAT...")

def apply_quantization(layer):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization,
)

print("Annotated model has been created successfully.")

annotated_model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy', F1_Score])


# annotated_model.summary()

annotated_model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=15,
    validation_data=valid_data,
    validation_steps=valid_data.samples // valid_data.batch_size
)

converter = tf.lite.TFLiteConverter.from_keras_model(annotated_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

tflite_model_path = "/path_to_save_directory/(QAT)RepViT_M11.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(quantized_tflite_model)
    
print("QAT model saved successfully.")

# --------------------------------------------- #
