import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import base64
from ultralyticsplus import YOLO, render_result  # Adjust accordingly if this import isn't correct

# Load the dataset
ds = load_dataset("keremberke/pothole-segmentation", name="full")

# Initialize the YOLO model
model = YOLO('keremberke/yolov8m-pothole-segmentation')
model.overrides['conf'] = 0.25
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 1000

# Set background with transparency
def set_background():
    with open("background.png", "rb") as file:
        bg_base64 = base64.b64encode(file.read()).decode('utf-8')
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-blend-mode: lighten;
            background-color: rgba(255, 255, 255, 0.75);  // Adjust transparency here
        }}
        </style>
        """, unsafe_allow_html=True
    )

set_background()

st.title("TECHNICAL ASSESSMENT for StreetScan (CityLogix)")
st.subheader("yolov8m pothole segmentation")
st.caption("Developed by Javad")

# Initialize selected_image
selected_image = None

# Sidebar for image selection with previews
st.sidebar.header("Select an Image")
for i, item in enumerate(ds['train']):
    image_data = item['image']
    if isinstance(image_data, str):
        response = requests.get(image_data)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_data) if not isinstance(image_data, Image.Image) else image_data
    if st.sidebar.button(f"Image {i+1}", key=i):
        selected_image = image
        st.sidebar.image(selected_image, caption=f"Selected Image {i+1}", width=100)

# Main area for image analysis
if st.button('Analyze Image'):
    if selected_image is not None:
        with st.spinner('Analyzing...'):
            results = model.predict(selected_image)
            render = render_result(model=model, image=selected_image, result=results[0])
            st.image(render, caption="Detection Results", use_column_width=True)
            st.write("Detection Boxes (x1, y1, x2, y2, confidence, class):", results[0].boxes)
            st.write("Masks (if available):", results[0].masks)
    else:
        st.error("Please select an image first.")
