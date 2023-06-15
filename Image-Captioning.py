#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers


# In[ ]:



import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Initialize the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Resize the image to fit the window
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image

        captions = predict_step(file_path)
        captions_text.set("\n".join(captions))


# Create the Tkinter window
window = tk.Tk()
window.title("Image Captioning")
window.geometry("400x400")

# Create the image label
image_label = tk.Label(window)
image_label.pack(pady=10)

# Create the button to select an image
select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Create the label to display captions
captions_text = tk.StringVar()
captions_label = tk.Label(window, textvariable=captions_text, wraplength=300)
captions_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()


# In[ ]:




