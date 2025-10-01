"""
VGG and ResNet Model Builder
Creates VGG and ResNet architectures for MG classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def build_model(architecture='vgg11_bn', num_classes=5, in_channels=1, pretrained=False):
    print("Building model with architecture: ", architecture)
    """
    Build a model with specified architecture (supports both VGG and ResNet).
    
    Args:
        architecture (str): Model architecture name (e.g., 'vgg11_bn', 'resnet18', 'resnet50', etc.)
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels
        pretrained (bool): Whether to use pretrained weights (uses IMAGENET1K_V1 weights if True)
    """
    # Get the model class dynamically
    model_class = getattr(models, architecture)
    
    # Use weights parameter instead of pretrained (for newer torchvision versions)
    if pretrained:
        model = model_class(weights='IMAGENET1K_V1')
    else:
        model = model_class(weights=None)
    
    # Check if it's a VGG model or ResNet model
    if hasattr(model, 'features'):
        # VGG model - modify first conv layer in features
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Modify final classifier layer for different number of classes
        # Get the last layer index dynamically
        last_layer_idx = len(model.classifier) - 1
        in_features = model.classifier[last_layer_idx].in_features
        model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)
        
    elif hasattr(model, 'conv1'):
        # ResNet model - modify first conv layer
        if in_channels != 3:
            # Get the original conv1
            conv1 = model.conv1
            model.conv1 = nn.Conv2d(in_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                                    stride=1, padding=conv1.padding, bias=conv1.bias)
        
        # Change the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    
    return model


def build_vgg_model(architecture='vgg11_bn', num_classes=5, in_channels=1, pretrained=False):
    """
    Build a VGG model with specified architecture.
    
    Args:
        architecture (str): 'vgg11_bn', 'vgg16_bn', 'vgg19_bn', etc.
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels
        pretrained (bool): Whether to use pretrained weights (uses IMAGENET1K_V1 weights if True)
    """
    return build_model(architecture, num_classes, in_channels, pretrained)


def build_vgg11_bn(num_classes=5, in_channels=1):
    """Backward compatibility function"""
    return build_vgg_model('vgg11_bn', num_classes, in_channels)


def build_vgg11(num_classes=5, in_channels=1):
    """Backward compatibility function"""
    return build_vgg_model('vgg11', num_classes, in_channels)


def build_vgg16_bn(num_classes=5, in_channels=1):
    """Build VGG16 with batch normalization"""
    return build_vgg_model('vgg16_bn', num_classes, in_channels)


def build_vgg16(num_classes=5, in_channels=1):
    """Build VGG16 with batch normalization"""
    return build_vgg_model('vgg16', num_classes, in_channels)


def build_vgg19(num_classes=5, in_channels=1):
    """Build VGG19 with batch normalization"""
    return build_vgg_model('vgg19', num_classes, in_channels)


def build_vgg19_bn(num_classes=5, in_channels=1):
    """Build VGG19 with batch normalization"""
    return build_vgg_model('vgg19_bn', num_classes, in_channels)
