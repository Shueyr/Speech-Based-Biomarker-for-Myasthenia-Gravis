"""
Model Inference Module
Handles loading and running inference with trained VGG/ResNet models.
"""

import torch
import torch.nn.functional as F
import os
import sys


class ModelInference:
    """Runs inference using trained classification models."""
    
    def __init__(self):
        """Initialize model inference handler."""
        # Add models directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(current_dir, "models")
        
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        
        # Import model builder with explicit path
        import importlib.util
        model_file_path = os.path.join(models_path, "model.py")
        print(f"Looking for model file at: {model_file_path}")
        print(f"File exists: {os.path.exists(model_file_path)}")
        
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found at: {model_file_path}")
            
        spec = importlib.util.spec_from_file_location("model_builder", model_file_path)
        model_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_builder)
        self.build_model = model_builder.build_model
        
        # Cache for loaded models
        self.loaded_models = {}
        
        print("Model inference handler initialized")
    
    def load_model(self, model_path, architecture, num_classes, in_channels=1):
        """
        Load a trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            architecture: Model architecture name (e.g., 'vgg19_bn', 'resnet18')
            num_classes: Number of output classes
            in_channels: Number of input channels (default: 1)
            
        Returns:
            Loaded PyTorch model in evaluation mode
        """
        # Check cache
        cache_key = f"{model_path}_{architecture}_{num_classes}"
        if cache_key in self.loaded_models:
            print(f"Using cached model: {architecture}")
            return self.loaded_models[cache_key]
        
        # Build model architecture
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.build_model(
            architecture=architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=False
        )
        
        # Load checkpoint
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        model.to(device)
        
        # Cache the model
        self.loaded_models[cache_key] = model
        
        print(f"Model loaded successfully: {architecture} with {num_classes} classes")
        
        return model
    
    def predict(self, input_tensor, model_path, architecture, num_classes, class_labels):
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Input tensor of shape [1, 257, 64]
            model_path: Path to model checkpoint
            architecture: Model architecture name
            num_classes: Number of output classes
            class_labels: List of class label strings
            
        Returns:
            Tuple of (predicted_class_index, confidence, label_string)
        """
        # Load model
        model = self.load_model(model_path, architecture, num_classes)
        device = next(model.parameters()).device
        
        # Ensure input tensor has correct shape [batch, channels, freq, time]
        if input_tensor.dim() == 3:
            # Has shape [1, 257, 64] - need to add channel dimension
            input_tensor = input_tensor.unsqueeze(1)  # [1, 1, 257, 64]
        elif input_tensor.dim() == 2:
            # Add batch and channel dimensions
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        
        # Move to device
        input_tensor = input_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
        
        # Get label
        if 0 <= predicted_idx < len(class_labels):
            label = class_labels[predicted_idx]
        else:
            label = f"Class {predicted_idx}"
        
        return predicted_idx, confidence, label
    
    def clear_cache(self):
        """Clear cached models to free memory."""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

