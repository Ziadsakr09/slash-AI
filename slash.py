import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import Counter

#Format the printed result
def pluralize(name, count):
    
    if count == 1:
        return name
    else:
        return name + 's'
    
# Load a model
model = YOLO('yolov8n.pt')  # load yolov8 pretrained model 



# Streamlit Application
st.title("Image Component Detector")
st.write("Upload an image, and click the 'Analyze Image' button to detect components in the image.")

#Uploading Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Image"):
        st.write("Analyzing...")

        #Run the YOLO model
        results=model(image)
        
        components=[]

        #Loop over detected components
        for i in range(len(results[0].boxes.cpu().numpy().cls)):
            components.append(results[0].names[results[0].boxes.cpu().numpy().cls[i]])

        #Count the number of each detected component
        counts = Counter(components)
        formatted_result = ", ".join([f"{count} {pluralize(component, count)}" for component, count in counts.items()])
        st.write("Detected Components:")
        st.write(formatted_result)
    else:
        
        st.write("No components detected.")