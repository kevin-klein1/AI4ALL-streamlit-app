

import streamlit as st
from PIL import Image
import time
import torch
import torch.nn as nn
from torchvision import transforms, models

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .benign {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .malignant {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for tracking predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'correct_count' not in st.session_state:
    st.session_state.correct_count = 0
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0


@st.cache_resource
def load_model():
    """Load the trained ResNet18 model."""
    # Initialize ResNet18 architecture matching your training setup
    model = models.resnet18(weights=None)
    
    # Modify final layer for binary classification (single output logit)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('breast_cancer_resnet18_v2.pth', map_location=torch.device('cpu')))
        model.eval()
        return model, True
    except FileNotFoundError:
        return None, False


def preprocess_image(image):
    """Preprocess the uploaded image for model inference."""
    # Same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)


def predict(model, image_tensor):
    """Run inference on the preprocessed image."""
    with torch.no_grad():
        output = model(image_tensor)  # Single logit output
        probability = torch.sigmoid(output).item()  # Convert to probability
        
        # Class 0 = Benign, Class 1 = Malignant (ImageFolder alphabetical order)
        predicted_class = 1 if probability > 0.5 else 0
        confidence = probability if predicted_class == 1 else (1 - probability)
    
    return predicted_class, confidence


# Load model
model, model_loaded = load_model()

# Header
st.markdown("<h1 class='main-header'> Breast Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>ResNet-based Histopathology Image Classification</h3>", unsafe_allow_html=True)

# Model status
if model_loaded:
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model file 'breast_cancer_resnet18_v2.pth' not found. Please place it in the same directory as app.py.")

# Introduction section
with st.expander("📖 About This Model"):
    st.markdown("""
    ### Background
    This application uses a **ResNet (Residual Neural Network)** deep learning model trained to classify 
    breast cancer histopathology images as either **benign** or **malignant**.
    
    ### Dataset
    The model was trained on the **BreaKHis 400X** dataset from Kaggle, which contains microscopic 
    biopsy images of breast tumor tissue at 400X magnification. This dataset includes:
    - **Benign tumors**: Non-cancerous growths that do not spread
    - **Malignant tumors**: Cancerous cells that can invade surrounding tissue
    
    ### Purpose
    Early and accurate detection of breast cancer is crucial for effective treatment. This tool 
    demonstrates how deep learning can assist pathologists in analyzing histopathology slides, 
    potentially improving diagnostic speed and consistency.
    
    ### How to Use
    1. Upload a histopathology image from the test set
    2. Select the true label (ground truth) for the image
    3. Click "Run Prediction" to see the model's classification
    4. Track the model's accuracy across multiple predictions
    """)

st.markdown("---")

# Accuracy tracker display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Predictions", st.session_state.total_count)
with col2:
    st.metric("Correct Predictions", st.session_state.correct_count)
with col3:
    accuracy = (st.session_state.correct_count / st.session_state.total_count * 100) if st.session_state.total_count > 0 else 0
    st.metric("Accuracy", f"{accuracy:.1f}%")

st.markdown("---")

# Image upload section
st.subheader("📤 Upload Test Image")

uploaded_file = st.file_uploader(
    "Drag and drop a breast histopathology image here",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
    help="Upload an image from the BreaKHis test set"
)

# True label selection
true_label = st.selectbox(
    "Select the true label (ground truth) for this image:",
    options=["Benign", "Malignant"],
    index=0,
    help="This is the actual classification from the test set"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown(f"""
        **Image Details:**
        - Filename: {uploaded_file.name}
        - Size: {image.size[0]} x {image.size[1]} pixels
        - Format: {image.format or 'Unknown'}
        - True Label: **{true_label}**
        """)
    
    # Prediction button
    if st.button("🔍 Run Prediction", type="primary", use_container_width=True):
        if not model_loaded:
            st.error("Cannot run prediction - model not loaded.")
        else:
            # Show loading spinner
            with st.spinner("Analyzing image..."):
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                # Get prediction from model
                predicted_class, confidence = predict(model, image_tensor)
                
                # Map class index to label
                predicted_label = "Malignant" if predicted_class == 1 else "Benign"
            
            # Check if prediction matches true label
            is_correct = predicted_label == true_label
            
            # Update session state
            st.session_state.total_count += 1
            if is_correct:
                st.session_state.correct_count += 1
            
            st.session_state.predictions.append({
                'filename': uploaded_file.name,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                box_class = "benign" if predicted_label == "Benign" else "malignant"
                st.markdown(f"""
                <div class='result-box {box_class}'>
                    <h3>Model Prediction</h3>
                    <h2>{predicted_label}</h2>
                    <p>Confidence: {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                box_class = "benign" if true_label == "Benign" else "malignant"
                st.markdown(f"""
                <div class='result-box {box_class}'>
                    <h3>True Label</h3>
                    <h2>{true_label}</h2>
                    <p>(Ground Truth)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show match result
            if is_correct:
                st.success("✅ **Correct!** The model prediction matches the true label.")
            else:
                st.error("❌ **Incorrect!** The model prediction does not match the true label.")
            
            # Force rerun to update metrics
            st.rerun()

# Prediction history
if st.session_state.predictions:
    st.markdown("---")
    st.subheader("📜 Prediction History")
    
    for i, pred in enumerate(reversed(st.session_state.predictions[-10:]), 1):
        status = "✅" if pred['correct'] else "❌"
        st.markdown(f"{status} **{pred['filename']}** | True: {pred['true_label']} | Predicted: {pred['predicted_label']} ({pred['confidence']*100:.1f}%)")
    
    if len(st.session_state.predictions) > 10:
        st.caption(f"Showing last 10 of {len(st.session_state.predictions)} predictions")
    
    # Reset button
    if st.button("🔄 Reset Statistics"):
        st.session_state.predictions = []
        st.session_state.correct_count = 0
        st.session_state.total_count = 0
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <p> <strong>Disclaimer:</strong> This tool is for educational and demonstration purposes only. 
    It should not be used for actual medical diagnosis.</p>
    <p>Model trained on BreaKHis 400X Dataset | Built with Streamlit & PyTorch</p>
</div>
""", unsafe_allow_html=True)