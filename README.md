# Image Captioning Project
This repository contains the code and resources for an image captioning project using PyTorch and pre-trained models such as VisionEncoderDecoderModel, ViTImageProcessor, and AutoTokenizer.

# Overview
The goal of this project is to generate descriptive captions for input images using deep learning techniques. The model architecture leverages the power of pre-trained models to extract image features and generate captions based on those features. The implementation utilizes the PyTorch framework along with the following pre-trained models:

VisionEncoderDecoderModel: This model combines an image encoder and a caption decoder. The image encoder processes the input image to extract high-level features, while the caption decoder generates a textual description based on these features.

ViTImageProcessor: The Vision Transformer (ViT) image processor is responsible for transforming the input image into a format compatible with the VisionEncoderDecoderModel. It performs necessary preprocessing steps such as resizing, normalization, and augmentation.

AutoTokenizer: The AutoTokenizer module automatically selects the appropriate tokenizer based on the selected pre-trained model. It handles the tokenization of text data, converting it into numerical representations that can be processed by the model.
