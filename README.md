# Lightweight Vision Transformers for medical diagnostics on the edge

This  repo contains the necessary code and instructions for developing quantized Vision Transformer (ViT)-based models for lightweight deployment on edge devices. 

This is the official Keras implemntation of our paper, **"On-Edge Deployment of Vision Transformers for Medical
Diagnostics: A Study on the Kvasir-Capsule Dataset,"** published in MDPI Applied Sciences (accessible through _____).

![alt text](https://github.com/DaraVaram/Mobile-ViT-Kvasir/blob/main/Pipeline.jpg)

## Introduction 
We present the quantization of ViT-based models with two approaches: 
1. Post-training quantization (PTQ); and
2. Quantization-aware training (QAT).

Models are initially trained and converted to .tflite files. The base models by themselves are float-32, whilst they can be quantized to float-16 and int-8 in PTQ. They can also separately be quantized with QAT, which experimentally generally performs the best in terms of size reduction and performance. 

The model architecture can be found below. In particular, the base mdoels used will be the following: 
1. EfficientFormerV2S2
2. EfficientViT_B0
3. EfficientViT_M4
4. MobileViT_V2_050
5. MobileViT_V2_100
6. MobileViT_V2_175
7. RepViT_M11

Although these were the models chosen for this particular study, you are not restricted in using these. The code is set up to be usable with the [Keras CV Attention Models](https://github.com/leondgarse/keras_cv_attention_models) repository. Further instructions on how this can be done is included further down. Other model parameters can also be modified as necessary.

![alt_text](https://github.com/DaraVaram/Mobile-ViT-Kvasir/blob/main/Architecture.jpg)

## Installation
The requirements.txt contains all the necessary libraries and packages for running the code and replicating the results.

## Model preparation:
The model_preparation.py file contains the actual creation and training of the model (and storing as .tflite files). 

The dataset used in this paper is the [Kvasir-Capsule](https://datasets.simula.no/kvasir-capsule/) dataset. The raw images have been undersampled to 500 images per class (for the top 9 classes) and divided further subsequently for training, testing and validation. Each of them have been put in separate directories (and loaded in which ImageDataGen) for convenience.

The model itself is built with the following code (corresponding to the architecture presented in the figure above): 

```python
base_model = repvit.RepViT_M11(input_shape = (224, 224, 3), num_classes = 0)
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
```

Here, you can change the base model (```base_model = repvit.RepViT_M11(input_shape = (224, 224, 3), num_classes = 0)```) depending on which of the Keras CV attention models you are using. They must first be imported through ``` from keras_cv_attention_models import ...```. You can also change other parameters such as the number of dense layers, the dropout, additional neurons, etc..., and the base model is **not trainable**.

## Training, testing and saving the model(s)
The training and testing of the model is straightforward and is presented below: 
```python
num_epochs = 15
K.clear_session()
history = model.fit(train_data, steps_per_epoch=train_steps_per_epoch,
                    validation_data=valid_data, validation_steps=valid_steps_per_epoch,
                    epochs=num_epochs)
```

```python
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
```

The models can be saved through the TFlite converter. The default conversion (with no quantization) can be done as follows: 
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('/path_to_your_save_directory/(F32)RepViT_M11.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("tflite_model successfully saved (float32 default model).")
    
print("-"*100)
```

To quantize to float-16: 
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()
with open('/path_to_your_save_directory/(F16)RepViT_M11.tflite', 'wb') as f:
    f.write(tflite_fp16_model)


print("tflite_model successfully saved (float16 quantized model).")

print("-"*100)
```

To quantize to int-8, representative data must be used: 
```python
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
with open('/path_to_your_save_directory/(I8)RepViT_M11.tflite', 'wb') as f:
    f.write(tflite_int8_model)

print("tflite_model successfully saved (int8 quantized model).")
print("-"*100)
```

## Quantization-aware training (QAT)
QAT can be performed by cloning the model using built-in keras functions. In particular, before the model is ready to be re-trained, the following lines of code are necessary:
```python
def apply_quantization(layer):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization,
)
```
Once the model has been cloned and annotated, it can be re-trained and converted to .tflite. 
```python
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

tflite_model_path = "/path_to_your_save_directory/RepViT_M11 Results/(QAT)RepViT_M11.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(quantized_tflite_model)
```

## Testing the performance of the models
Models can be tested in a variety of different ways. We will, however, just look at the accuracy and F1-score for the purposes of this demonstration. Once the models have been saved to a particular directory (as .tflite files), we can test them with the following function: 
```python
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
```

The .tflite models have to be loaded in (and interpreted) to be able to get the performance metrics of the model. 
```python
test_data = "path/to/your/test/data/"
interpreter = tf.lite.Interpreter(model_path= "/path/to/your/tflite/model")
interpreter.allocate_tensors()

accuracy, f1 = evaluate_tflite_model_metrics(interpreter, test_data)
```

The code in its full-form can be found through ```model_preparation.py``` and ```model_testing.py```. 

If you found our work useful or helpful for your own research, please consider citing us using the below: 
- ### BibTeX:
```

```
