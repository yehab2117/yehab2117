import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Placeholder for a dictionary mapping detected objects to their carbohydrate content
carb_content = {
    'Hamburger': 30,
    'pizza': 40,
    'apple': 25,
    'banana': 27,
    'water melon': 22
    # Add more mappings as needed
}

calories_content = {
    'Hamburger': 30,
    'pizza': 40,
    'apple': 25,
    'banana': 27,
    'water melon': 22
    # Add more mappings as needed
}

# Placeholder for your class names
class_names = ['burger', 'pizza', 'water melon', 'Tomato', 'banana', 'French fries', 'mango', 'Carrot', 'Pepsi', 'undefined']  # Add all your class names

def detect_objects(image, model_path):
    model = YOLO(model_path)
    prediction = model.predict(image)
    detected_labels = []
    for box in prediction[0].boxes:
        class_id = int(box.cls[0])  # Extract class id
        class_name = class_names[class_id]  # Map class id to class name
        detected_labels.append(class_name)
    image = prediction[0].plot()
    return image, detected_labels

def calculate_total_carbs(detected_labels, carb_content):
    total_carbs = 0
    for label in detected_labels:
        if label in carb_content:
            total_carbs += carb_content[label]
    return total_carbs

def calculate_total_calories(detected_labels, calories_content):
    total_calories = 0
    for label in detected_labels:
        if label in calories_content:
            total_calories += calories_content[label]
    return total_calories

st.title("Object Detection and Carbohydrate Counting App")
st.write("Upload an image for detection:")

# Add a selectbox to choose between fast food and vegetables
option = st.selectbox(
    'Choose the category for detection:',
    ('Fast Food', 'Vegetables', 'Fruit', 'Product')
)

# Map options to model file paths
model_paths = {
    'Fast Food': r"C:\Users\2021\Desktop\fast.pt",
    'Vegetables': r"C:\Users\2021\Desktop\vegtabels.pt",
    'Fruit': r"C:\Users\2021\Desktop\fast.pt",
    'Product': r"C:\Users\2021\Desktop\fast.pt"
}

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Get the model path based on the selection
    model_path = model_paths[option]
    
    # Perform detection using the selected model
    detected_image, detected_labels = detect_objects(image, model_path)
    
    # Convert the image from BGR to RGB
    detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    
    st.image(detected_image_rgb, caption="Image with Detected Objects", use_column_width=True)
    
    # Calculate total carbohydrates and calories
    total_carbs = calculate_total_carbs(detected_labels, carb_content)
    total_calories = calculate_total_calories(detected_labels, calories_content)
    
    # Display carbohydrate and calorie content
    st.write(f"Total Carbohydrates: {total_carbs} grams")
    st.write(f"Total Calories: {total_calories} calories")
    
    # Optionally, display detected labels and their corresponding carb and calorie values
    st.write("Detected items and their carbohydrate and calorie content:")
    for label in detected_labels:
        if label in carb_content:
            st.write(f"{label}: {carb_content[label]} grams")
    for label in detected_labels:
        if label in calories_content:
            st.write(f"{label}: {calories_content[label]} calories")
