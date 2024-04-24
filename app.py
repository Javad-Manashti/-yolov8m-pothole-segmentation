import streamlit as st
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import base64
from ultralyticsplus import YOLO, render_result

# Set the page title and layout
st.set_page_config(page_title="Yolov8m Pothole Segmentation", layout="wide")

st.title("Yolov8m Pothole Segmentation")

st.caption("Developed by Javad")


# Assuming the background image is saved at the path 'background.png'
def set_background():
    with open("background.png", "rb") as file:
        bg_base64 = base64.b64encode(file.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center center;
            background-blend-mode: darken;
            background-color: rgba(0, 0, 0, 0.75);
        }}
        /* Narrower sidebar */
        .css-1d391kg {{
            padding-top: 0rem;
            padding-right: 1rem;
            padding-bottom: 10rem;
            padding-left: 1rem;
            background-color: rgba(255, 255, 255, 0.5); /* Set sidebar background color and transparency */
            width: 150px; /* Adjust the width of the sidebar */
        }}
        /* Image buttons in the sidebar */
        .stButton > button {{
            display: block;
            width: 100%;
            margin-bottom: 5px;
            border: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background()

# load model
model = YOLO('keremberke/yolov8n-pothole-segmentation')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image


# Corrected function to load images from dataset
def load_image_from_dataset(image_data):
    if isinstance(image_data, str) and image_data.startswith("http"):
        response = requests.get(image_data)
        image = Image.open(BytesIO(response.content))
    elif isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        image = Image.open(image_data)
    return image
# Load the dataset
ds = load_dataset("keremberke/pothole-segmentation", name="full")

# Main area for image analysis and display
st.sidebar.header("Select an Image")

image_index = st.sidebar.selectbox(
    "Choose an image:",
    range(len(ds['train'])),
    format_func=lambda x: f"Image {x+1}"
)
image_data = ds['train'][image_index]['image']
selected_image = load_image_from_dataset(image_data)


# Main area for image display and analysis
if selected_image is not None:
    # Display the raw image at the top of the main page
    st.image(selected_image, caption=f"Raw Image {image_index + 1}", use_column_width=True)

    # Analyze button in the sidebar
    # if st.sidebar.button('Analyze Image', key="analyze_button"): # Unique key for the button
    with st.spinner('Analyzing...'):
        results = model.predict(selected_image)
        if results and results[0].boxes is not None:
            render = render_result(model=model, image=selected_image, result=results[0])
            # Display the analysis results below the raw image
            st.image(render, caption="Detection Results", use_column_width=True)
        else:
            st.error("No detections were found.")
 


# Function to format and print detection results
def print_detections(detections, masks):
    if detections is None or len(detections) == 0:
        st.error("No detections found", unsafe_allow_html=True)
    else:
        # Loop through each detection and print formatted results
        for det in detections:
            # Each 'det' has (x1, y1, x2, y2, confidence, class)
            x1, y1, x2, y2, conf, class_id = det
            st.markdown(
                f"**Detection:** Class ID: `{int(class_id)}`, Confidence: `{conf:.2f}`, "
                f"Bounding box: `[{int(x1)}, {int(y1)}], [{int(x2)}, {int(y2)}]`"
            )
    # Check if there are any mask data available
    if masks is not None and hasattr(masks, "xy") and len(masks.xy) > 0:
        st.markdown("**Masks available.**")
    else:
        st.markdown("**No masks available.**")

 
