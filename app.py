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
    img = img / 255.0  # Normalize to [0, 1]
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

# Non-Maximum Suppression (NMS)
def nms(boxes, scores, iou_threshold=0.5):
    indices = tf.image.non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size=50,
        iou_threshold=iou_threshold
    )
    return indices.numpy()

# Function to get top predictions after applying NMS
def get_top_predictions(classes, scores, boxes, label_map, threshold=0.5, iou_threshold=0.5):
    indices = nms(boxes, scores, iou_threshold=iou_threshold)
    predictions = []
    for i in indices:
        if scores[i] >= threshold:
            class_id = int(classes[i])
            label = label_map.get(class_id, 'Unknown')
            predictions.append((label, scores[i], boxes[i]))
    return predictions

# Function to draw boxes and labels on the image
def draw_boxes(image, boxes, classes, scores, label_map, threshold=0.5, iou_threshold=0.5):
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    predictions = get_top_predictions(classes, scores, boxes, label_map, threshold, iou_threshold)
    
    for label, score, box in predictions:
        ymin, xmin, ymax, xmax = box
        ymin = int(max(1, ymin * height))
        xmin = int(max(1, xmin * width))
        ymax = int(min(height, ymax * height))
        xmax = int(min(width, xmax * width))
        
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
            label, score, box = detection
            ymin, xmin, ymax, xmax = box
            f.write(f'{label} {score:.4f} {ymin} {xmin} {ymax} {xmax}\n')

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
    
    # Annotate image with NMS
    annotated_image = draw_boxes(image, boxes, classes, scores, label_map)
    st.image(annotated_image, caption='Annotated Image', use_column_width=True)
    
    # Save results to text file
    savepath = 'save/results'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    predictions = get_top_predictions(classes, scores, boxes, label_map)
    save_results_to_txt(predictions, savepath, uploaded_file.name)
    
    st.header('Predicted Medications:')
    for label, score, _ in predictions:
        st.write(f"_{label}_: {score * 100:.2f}%")
