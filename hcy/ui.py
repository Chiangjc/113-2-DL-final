import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple
import io
import os


class ResNetMultilabel(nn.Module):
    """基於 ResNet 的 multi-label 分類模型"""
    def __init__(self, num_classes, pretrained=True, model_name='resnet50', dropout_rate=0.5):
        super(ResNetMultilabel, self).__init__()

        # 載入預訓練的 ResNet 變體
        if model_name == 'resnet18':
            backbone = torchvision.models.resnet18(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet34':
            backbone = torchvision.models.resnet34(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            num_features = 2048
        elif model_name == 'resnet101':
            backbone = torchvision.models.resnet101(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"不支援的模型: {model_name}")

        # 移除原本分類層，只保留到 avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # 新增自訂的 multi-label 分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        print(f"模型初始化完成: {model_name}, 預訓練: {pretrained}")
        print(f"特徵維度: {num_features}, 類別數: {num_classes}")

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        output = self.classifier(features)
        return output



# Configure page
st.set_page_config(
    page_title="Chiikawa Character Detector",
    page_icon="🐾",
    layout="wide"
)

# Define your character labels - update this list to match your models
CHARACTER_LABELS = [
    "Chiikawa","Hachiware",  "kurimanju", "Momonga", "Rakko", "Shisa", "Usagi"
]

@st.cache_resource
def load_model(model_name: str):
    """
    Load PyTorch model. Update the paths and model architectures to match your setup.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model paths - update these paths to your actual model files
    model_paths = {
        "ResNet50": "./resnet50_bs32_lr0.0005_ep30_20250531_145459/best_multilabel_model.pt",
        "EfficientNet-B0": "models/efficientnet_chiikawa.pth", 
        "Vision Transformer": "models/vit_chiikawa.pth",
        "MobileNetV2": "models/mobilenet_chiikawa.pth",
        "Custom CNN": "models/custom_cnn_chiikawa.pth"
    }
    
    try:
        model_path = model_paths.get(model_name)
        if model_path and os.path.exists(model_path):
            # Load your trained model
            # Example for a generic multilabel classifier:
            model = ResNetMultilabel(
                    num_classes=7,
                    pretrained=True,
                    model_name="resnet50",
                    dropout_rate=0.5
                ).to("cuda")
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # If you saved just the state dict, use this instead:
            # model = YourModelClass(num_classes=len(CHARACTER_LABELS))
            # model.load_state_dict(torch.load(model_path, map_location=device))
            
            model.eval()
            model.to(device)
            return model, device
        else:
            st.error(f"Model file not found: {model_path}")
            return None, device
            
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None, device

def get_image_transforms():
    """
    Define image preprocessing transforms.
    Adjust these based on how you trained your models.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the uploaded image for PyTorch model prediction.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_image_transforms()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_characters(model, image_tensor: torch.Tensor, device, threshold: float = 0.5) -> Dict[str, float]:
    """
    Make predictions using the PyTorch model.
    """
    if model is None:
        # Return mock predictions if model failed to load
        import random
        predictions = {}
        for label in CHARACTER_LABELS:
            prob = random.uniform(0.1, 0.9)
            predictions[label] = prob
        return predictions
    
    try:
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor)
            
            # Apply sigmoid for multilabel classification
            probabilities = torch.sigmoid(outputs)
            
            # Convert to numpy and create dictionary
            probs_np = probabilities.cpu().numpy().flatten()
            
            predictions = {}
            for i, label in enumerate(CHARACTER_LABELS):
                if i < len(probs_np):
                    predictions[label] = float(probs_np[i])
                else:
                    predictions[label] = 0.0
                    
        return predictions
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return {label: 0.0 for label in CHARACTER_LABELS}

def main():
    st.title("🐾 Chiikawa Character Detector")
    st.markdown("Upload an image to detect Chiikawa characters using your trained models!")
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Selection")
        available_models = [
            "ResNet50",
            "EfficientNet-B0",
            "Vision Transformer",
            "MobileNetV2",
            "Custom CNN"
        ]
        
        selected_model = st.selectbox(
            "Choose a model:",
            available_models,
            help="Select which trained model to use for character detection"
        )
        
        # Detection threshold
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Minimum confidence score to display a detected character"
        )
        
        # Load selected model
        with st.spinner(f"Loading {selected_model} model..."):
            model, device = load_model(selected_model)
        
        if model is not None:
            st.success(f"✅ {selected_model} loaded successfully!")
            st.info(f"🖥️ Using device: {device}")
        else:
            st.error(f"❌ Failed to load {selected_model}")
            st.warning("Using demo mode with mock predictions")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload an image containing Chiikawa characters"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📏 Image size: {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        st.header("🔍 Detection Results")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Make predictions
                predictions = predict_characters(model, processed_image, device, confidence_threshold)
                
                # Filter predictions above threshold
                detected_characters = {
                    char: conf for char, conf in predictions.items() 
                    if conf >= confidence_threshold
                }
                
                if detected_characters:
                    st.success(f"🎉 Found {len(detected_characters)} character(s)!")
                    
                    # Sort by confidence
                    sorted_detections = sorted(
                        detected_characters.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    # Display results as metrics
                    for char, conf in sorted_detections:
                        st.metric(
                            label=char,
                            value=f"{conf:.1%}",
                            help=f"Confidence: {conf:.3f}"
                        )
                    
                    # Create results dataframe
                    results_df = pd.DataFrame([
                        {"Character": char, "Confidence": f"{conf:.1%}", "Score": conf}
                        for char, conf in sorted_detections
                    ])
                    
                    st.subheader("📊 Detailed Results")
                    st.dataframe(
                        results_df[["Character", "Confidence"]], 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Progress bars for visual representation
                    st.subheader("📈 Confidence Levels")
                    for char, conf in sorted_detections:
                        st.write(f"**{char}**")
                        st.progress(conf)
                        st.write("")  # Add spacing
                        
                else:
                    st.warning(f"No characters detected above {confidence_threshold:.1%} confidence threshold.")
                    st.info("Try lowering the confidence threshold in the sidebar.")
        
        else:
            st.info("👆 Please upload an image to start detection")
    
    # Additional information
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        This Streamlit app uses trained PyTorch multilabel classification models to detect Chiikawa characters in images.
        
        **Features:**
        - Multiple PyTorch model support
        - GPU acceleration when available
        - Adjustable confidence threshold
        - Real-time predictions with sigmoid activation
        - Detailed results with confidence scores
        
        **How to use:**
        1. Select your preferred model from the sidebar
        2. Adjust the confidence threshold if needed
        3. Upload an image containing Chiikawa characters
        4. View the detection results with confidence scores
        
        **Model Requirements:**
        - Models should be saved as `.pth` files
        - Expected input: 224x224 RGB images (adjustable)
        - Output: Raw logits for multilabel classification
        - Sigmoid activation applied automatically
        
        **Setup Instructions:**
        1. Update `model_paths` dictionary with your model file paths
        2. Modify `CHARACTER_LABELS` list to match your classes
        3. Adjust image preprocessing transforms if needed
        4. Update model architecture loading if using custom models
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**PyTorch Configuration:**")
            st.write(f"- PyTorch Version: {torch.__version__}")
            st.write(f"- CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"- GPU Device: {torch.cuda.get_device_name()}")
        
        with col2:
            st.write("**Image Processing:**")
            st.write("- Input Size: 224×224 pixels")
            st.write("- Normalization: ImageNet mean/std")
            st.write("- Activation: Sigmoid for multilabel")
    
    # Model comparison (if multiple images uploaded)
    if st.checkbox("🔄 Enable Model Comparison Mode"):
        st.subheader("Model Comparison")
        st.info("Upload the same image multiple times with different models to compare results!")

if __name__ == "__main__":
    main()