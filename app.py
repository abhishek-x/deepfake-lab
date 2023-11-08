import streamlit as st
import cv2
import numpy as np
import dlib
import tempfile
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from cv2 import applyColorMap, COLORMAP_JET

# Constants and configurations
FACE_DETECTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
MODEL_PATH = 'model.h5'
ROI_WIDTH = 40
ROI_LENGTH = 20
PPG_MAP_SIZE = (128, 32)
VIDEO_TYPES = ["mp4", "avi"]

# Initialize face detector and predictor
face_detector = dlib.get_frontal_face_detector()
try:
    dlib_facelandmark = dlib.shape_predictor(FACE_DETECTOR_PATH)
except Exception as e:
    st.error(f"Error loading face predictor: {e}")
    st.stop()

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function definitions
def get_facial_landmarks(image):
    """Extract facial landmarks from the image using dlib."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image)
    if faces:
        return dlib_facelandmark(gray_image, faces[0])
    return None

def get_roi_region(image, x, y):
    """Get Region of Interest (ROI) around the given coordinates."""
    roi_x = x - ROI_WIDTH // 2
    roi_y = y - ROI_LENGTH // 2
    roi = image[roi_y:roi_y + ROI_LENGTH, roi_x:roi_x + ROI_WIDTH, 1]
    return roi

def flatten_roi(roi):
    """Flatten the ROI for processing."""
    roi_tensor = tf.convert_to_tensor(roi, dtype=tf.float32)
    roi_tensor = tf.reshape(roi_tensor, shape=(1, ROI_LENGTH, ROI_WIDTH, 1))
    max_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5))
    return tf.reshape(max_pool_2d(roi_tensor), shape=(1, PPG_MAP_SIZE[1]))

def generate_ppg_map_from_video(video_path):
    """Generate PPG map from the video."""
    cap = cv2.VideoCapture(video_path)
    ppg_map = np.zeros(PPG_MAP_SIZE)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    for count, frame in enumerate(frames[:PPG_MAP_SIZE[0]]):
        landmarks = get_facial_landmarks(frame)
        if landmarks:
            roi = get_roi_region(frame, landmarks.part(30).x, landmarks.part(30).y)
            ppg_map[count] = flatten_roi(roi).numpy()

    return ppg_map

def classify_image(image_np, model):
    """Classify the image using the model."""
    image_resized = cv2.resize(image_np, PPG_MAP_SIZE[::-1])  # Ensure correct dimensions
    image_normalized = image_resized / 255.0  # Normalize the pixel values

    predictions = model.predict(np.expand_dims(image_normalized, axis=0))
    print(predictions)
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(predicted_class)
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit UI components
st.title('Deepfake Video Recognition Lab 🔬')
st.write("This tool classifies PPG Maps of a Human Video as Original or Manipulated.")

uploaded_file = st.file_uploader("Upload a video...", type=VIDEO_TYPES)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:

        if st.button('Analyze Video'):
            st.subheader('Step 1: Uploading the video 🎬')
            tfile.write(uploaded_file.read())
            
            # Display video
            st.video(tfile.name)

            st.subheader('Step 2: Processing the video to generate PPG map 🗺️')
            ppg_map = generate_ppg_map_from_video(tfile.name)

            # Normalize and apply color map
            ppg_map_norm = cv2.normalize(ppg_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ppg_map_color = applyColorMap(ppg_map_norm, COLORMAP_JET)

            ppg_map_color_landscape = ppg_map_color
            ppg_map_color_landscape = np.rot90(ppg_map_color_landscape)
            st.image(ppg_map_color_landscape, caption='Generated PPG Map', use_column_width=True)

            st.subheader('Step 3: Classifying the video 👨‍🔬')
            with st.spinner('Classifying...'):
                label, confidence = classify_image(ppg_map_color, model)
                labels = {0: 'Fake', 1: 'Real'}
                if (label == 0):
                    st.error(f"❌ Prediction: Manipulated. Model Confidence: {confidence:.2f}")
                else:
                    st.success(f"✅ Prediction: Original. Model Confidence: {confidence:.2f}")
