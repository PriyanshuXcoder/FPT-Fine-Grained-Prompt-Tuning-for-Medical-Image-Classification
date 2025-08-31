import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.models import vit_b_16
from model_architecture import FineGrainedPrompt, FusionModule, LightweightNetwork
from data_preprocessing import get_data_loaders

def train_model(train_loader, model, feature_extractor, prompt_module, fusion_module, optimizer, criterion, num_epochs=10):
    model.train()
    feature_extractor.eval()  # Freeze the feature extractor
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                vit_features = feature_extractor(images)  # Extract ViT features
            prompt_embeddings = prompt_module(batch_size=images.size(0))  # Generate prompts
            prompt_embeddings = prompt_embeddings.to(device)
            # Fuse ViT features with prompts
            fused_features = fusion_module(vit_features, prompt_embeddings)
            # Pass fused features to the lightweight network
            outputs = model(fused_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_channels = 768  # Output size of ViT
    hidden_dim = 256
    num_layers = 6
    num_prompts = 5
    prompt_dim = 768

    # Initialize ViT model
    feature_extractor = vit_b_16(weights="IMAGENET1K_V1").to(device)
    feature_extractor.heads = nn.Identity()

    # Initialize fine-grained prompts and fusion module
    prompt_module = FineGrainedPrompt(num_prompts, prompt_dim).to(device)
    fusion_module = FusionModule(input_dim=768, hidden_dim=hidden_dim).to(device)

    # Initialize the lightweight network
    side_network = LightweightNetwork(hidden_dim, hidden_dim, num_layers).to(device)

    # Set up optimizer and loss function
    optimizer = AdamW(list(side_network.parameters()) + list(prompt_module.parameters()) + list(fusion_module.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Get data loaders
    data_dir = 'data/'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir)

    # Train the model
    train_model(train_loader, side_network, feature_extractor, prompt_module, fusion_module, optimizer, criterion)

    torch.save(side_network.state_dict(), 'models/vit_trained_model_with_prompts.pth')
    print("Model has been successfully saved as 'vit_trained_model_with_prompts.pth'.")





# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torchvision.models import vit_b_16  # Import Vision Transformer model
# from model_architecture import LightweightNetwork
# from data_preprocessing import get_data_loaders

# def train_model(train_loader, model, feature_extractor, optimizer, criterion, num_epochs=10):
#     model.train()
#     feature_extractor.eval()  # Freeze the feature extractor

#     for epoch in range(num_epochs):
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             # Extract features using ViT model
#             with torch.no_grad():
#                 features = feature_extractor(images)
#             # Forward pass through the lightweight network
#             outputs = model(features)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     input_channels = 768  # The output size of ViT's last layer
#     hidden_dim = 256
#     num_layers = 6

#     # Initialize Vision Transformer as feature extractor
#     feature_extractor = vit_b_16(weights="IMAGENET1K_V1").to(device)
#     feature_extractor.heads = nn.Identity()  # Remove the classification head

#     # Initialize the lightweight network
#     side_network = LightweightNetwork(input_channels, hidden_dim, num_layers).to(device)

#     # Set up optimizer and loss function
#     optimizer = AdamW(side_network.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()

#     # Get data loaders
#     data_dir = 'data/'
#     train_loader, val_loader, test_loader = get_data_loaders(data_dir)

#     # Train the model
#     train_model(train_loader, side_network, feature_extractor, optimizer, criterion)

#     torch.save(side_network.state_dict(), 'models/vit_trained_model.pth')
#     print("Model has been successfully saved as 'vit_trained_model.pth'.")


# # import torch
# # import torch.nn as nn
# # from torch.optim import AdamW
# # from torchvision.models import resnet18 
# # from model_architecture import LightweightNetwork
# # from data_preprocessing import get_data_loaders
# # from torchvision.models import ResNet18_Weights

# # def train_model(train_loader, model, feature_extractor, optimizer, criterion, num_epochs=10):
# #     model.train()
# #     feature_extractor.eval()  # Freeze the feature extractor
# #     for epoch in range(num_epochs):
# #         for images, labels in train_loader:
# #             optimizer.zero_grad()
# #             # Extract features using the pre-trained model
# #             with torch.no_grad():
# #                 features = feature_extractor(images)
# #             # Forward pass through the lightweight network
# #             outputs = model(features)
# #             loss = criterion(outputs, labels)
# #             loss.backward()
# #             optimizer.step()
# #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # if __name__ == '__main__':
# #     input_channels = 512  # The output size of ResNet18's last convolutional layer
# #     hidden_dim = 256
# #     num_layers = 6
# #     # Initialize feature extractor
# #     feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# #     feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])  # Remove the classifier layers
# #     # Initialize the lightweight network
# #     side_network = LightweightNetwork(input_channels, hidden_dim, num_layers)
# #     # Set up optimizer and loss function
# #     optimizer = AdamW(side_network.parameters(), lr=1e-3)
# #     criterion = nn.CrossEntropyLoss()
# #     # Get data loaders
# #     data_dir = 'data/'
# #     train_loader, val_loader, test_loader = get_data_loaders(data_dir)
# #     # Train the model
# #     train_model(train_loader, side_network, feature_extractor, optimizer, criterion)

# #     torch.save(side_network.state_dict(), 'models/fpt_plus_trained_model.pth')
# #     print("Model has been successfully saved as 'fpt_plus_trained_model.pth'.")
# # 6