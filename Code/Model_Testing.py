## Necessary imports ---------------------------------------------------------------- ##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img


import tensorflow_model_optimization as tfmot

import logging
import tempfile
import os
from sklearn.metrics import f1_score

## ---------------------------------------------------------------------------------- ##


test_dir = "/path/to/testing/set"
img_size = 224
batch_size = 32

datagen = ImageDataGenerator(
    rescale = 1./255 
)

test_data = datagen.flow_from_directory(
    test_dir, 
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle = False
)


def evaluate_tflite_model_metrics(interpreter, test_data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    input_dtype = interpreter.get_input_details()[0]['dtype']

    total_seen = 0
    total_correct = 0
    true_labels = []
    predicted_labels = []

    for images, labels in test_data:
        for i in range(images.shape[0]):
            input_data = images[i][np.newaxis, :, :, :]
            if input_dtype == np.uint8:
                input_data = (input_data * 255).astype(np.uint8)
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            predicted_label = np.argmax(output_data[0])
            true_label = np.argmax(labels[i])

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

            if predicted_label == true_label:
                total_correct += 1
            total_seen += 1

        if total_seen >= test_data.samples:
            break

    accuracy = total_correct / total_seen
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return accuracy, f1

interpreter = tf.lite.Interpreter(model_path= "/path/to/saved/tflite/models") ## You can replace this path with any of the quantized TFlite models (F16, I8, QAT, etc...)
interpreter.allocate_tensors()

accuracy, f1 = evaluate_tflite_model_metrics(interpreter, test_data)
print("Base model: ")
print("TFLite Model Testing Accuracy: {:.4f}%".format(accuracy * 100))
print("TFLite Model Testing F1-score: {:.4f}%".format(f1 *100))

print("-"*100)
