import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import streamlit as st
import matplotlib.pyplot as plt

# Function to load the label map from a file
def load_labelmap(labelmap_path):
    label_map = {}
    with open(labelmap_path, 'r') as file:
        for line in file:
            if ':' in line:
                id_str, name = line.strip().split(':')
                id = int(id_str)
                label_map[id] = name.strip()
    return label_map

# Function to preprocess the input image
def preprocess_image(image, height, width):
    img = image.resize((width, height))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# Function to perform prediction using TFLite model
def predict(interpreter, image, input_details, output_details):
    img = preprocess_image(image, input_details[0]['shape'][1], input_details[0]['shape'][2])
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores

    return boxes, classes, scores

# Function to get top predictions
def get_top_predictions(classes, scores, label_map, threshold=0.4):
    predictions = []
    for i, score in enumerate(scores):
        if score >= threshold:
            class_id = int(classes[i])  # Convert to integer ID
            label = label_map.get(class_id, 'Unknown')
            predictions.append((label, score))
    return predictions

# Function to draw boxes and labels on the image
def draw_boxes(image, boxes, classes, scores, label_map, threshold=0.5):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for i, score in enumerate(scores):
        if score >= threshold:
            class_id = int(classes[i])
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, ymin * height))
            xmin = int(max(1, xmin * width))
            ymax = int(min(height, ymax * height))
            xmax = int(min(width, xmax * width))
            
            label = label_map.get(class_id, 'Unknown')
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
            draw.text((xmin, ymin - 10), f"{label} {score*100:.2f}%", fill='red')
    
    return image

# Function to save detection results to a text file
def save_results_to_txt(detections, savepath, image_path):
    image_fn = os.path.basename(image_path)
    base_fn, ext = os.path.splitext(image_fn)
    txt_result_fn = base_fn + '.txt'
    txt_savepath = os.path.join(savepath, txt_result_fn)

    with open(txt_savepath, 'w') as f:
        for detection in detections:
            f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

# Streamlit app
st.title('Pills Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the TFLite model and allocate tensors
    model_path = "detect.tflite"
    labelmap_path = "labelmap.txt"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load label map
    label_map = load_labelmap(labelmap_path)
    
    # Process the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform prediction
    boxes, classes, scores = predict(interpreter, image, input_details, output_details)
    predictions = get_top_predictions(classes, scores, label_map)
    
    # Annotate image
    annotated_image = draw_boxes(image, boxes, classes, scores, label_map)
    st.image(annotated_image, caption='Annotated Image', use_column_width=True)
    
    # Save results to text file
    savepath = 'save/results'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    detections = []
    for i, score in enumerate(scores):
        if score >= 0.5:
            class_id = int(classes[i])
            ymin, xmin, ymax, xmax = boxes[i]
            detections.append([label_map.get(class_id, 'Unknown'), score, ymin, xmin, ymax, xmax])
    
    save_results_to_txt(detections, savepath, uploaded_file.name)
    
    st.header('Predicted Medications:')
    for label, score in predictions:
        st.write(f"_{label}_: {score * 100:.2f}%")
