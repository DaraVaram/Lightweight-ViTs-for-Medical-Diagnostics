# Mobile-ViT-Kvasir

This  repo contains the necessary code and instructions for developing quantized Vision Transformer (ViT)-based models for lightweight deployment on edge devices. 

This is the official Keras implemntation of our paper, **"On-Edge Deployment of Vision Transformers for Medical
Diagnostics: A Study on the Kvasir-Capsule Dataset,"** published in MDPI Applied Sciences (accessible through _____).

![alt text](https://github.com/DaraVaram/Mobile-ViT-Kvasir/blob/main/Pipeline.jpg)

## Introduction 
We present the quantization of ViT-based models with two approaches: 
1. Post-training quantization (PTQ); and
2. Quantization-aware training (QAT).

Models are initially trained and converted to .tflite files. The base models by themselves are float-32, whilst they can be quantized to float-16 and int-8 in PTQ. They can also separately 
