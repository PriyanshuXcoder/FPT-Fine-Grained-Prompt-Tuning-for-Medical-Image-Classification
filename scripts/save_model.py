# save_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from scripts.model_architecture import LightweightNetwork  # Updated import path

# Initialize feature extractor
feature_extractor = resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])  # Remove the classifier layers

# Initialize the lightweight network
input_channels = 512  # The output size of ResNet18's last convolutional layer
hidden_dim = 256
num_layers = 6
side_network = LightweightNetwork(input_channels, hidden_dim, num_layers)

# Load the trained model state
checkpoint_path = 'models/fpt_plus_trained_model.pth'  # Update this path if needed
side_network.load_state_dict(torch.load(checkpoint_path))

# Save the model
torch.save(side_network.state_dict(), 'models/fpt_plus_trained_model.pth')
print("Model saved successfully.")
