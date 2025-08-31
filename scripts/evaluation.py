import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from model_architecture import FineGrainedPrompt, FusionModule, LightweightNetwork
from torchvision.models import vit_b_16

def evaluate_model(model, feature_extractor, prompt_module, fusion_module, test_loader):
    model.eval()
    feature_extractor.eval()
    prompt_module.eval()
    fusion_module.eval()
    
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            vit_features = feature_extractor(images)
            prompt_embeddings = prompt_module(batch_size=images.size(0)).to(device)
            fused_features = fusion_module(vit_features, prompt_embeddings)
            outputs = model(fused_features)
            probs = F.softmax(outputs, dim=1)
            predictions.extend(probs[:, 1].cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predicted_classes = [1 if p > 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(true_labels, predicted_classes)
    auc_score = roc_auc_score(true_labels, predictions)
    print(f'Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}')

if __name__ == '__main__':
    from data_preprocessing import get_data_loaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_extractor = vit_b_16(weights="IMAGENET1K_V1").to(device)
    feature_extractor.heads = nn.Identity()

    input_channels = 768
    hidden_dim = 256
    num_layers = 6
    num_prompts = 5
    prompt_dim = 768

    prompt_module = FineGrainedPrompt(num_prompts, prompt_dim).to(device)
    fusion_module = FusionModule(input_dim=768, hidden_dim=hidden_dim).to(device)
    model = LightweightNetwork(hidden_dim, hidden_dim, num_layers).to(device)
    model.load_state_dict(torch.load('models/fpt_plus_trained_model.pth', map_location=device))

    data_dir = 'data/'
    _, _, test_loader = get_data_loaders(data_dir)
    evaluate_model(model, feature_extractor, prompt_module, fusion_module, test_loader)




# # evaluation.py
# import torch
# from sklearn.metrics import accuracy_score, roc_auc_score
# import torch.nn.functional as F
# from model_architecture import LightweightNetwork  # Import the correct architecture

# def evaluate_model(model, feature_extractor, test_loader):
#     model.eval()
#     feature_extractor.eval()
#     predictions = []
#     true_labels = []
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device)  # Move images to the appropriate device (CPU/GPU)
#             labels = labels.to(device)  # Move labels to the appropriate device (CPU/GPU)
#             features = feature_extractor(images)
#             outputs = model(features)
#             probs = F.softmax(outputs, dim=1)  # Get class probabilities
#             predictions.extend(probs[:, 1].cpu().numpy())  # Assuming class 1 is the positive class
#             true_labels.extend(labels.cpu().numpy())
#     # Calculate accuracy and AUC
#     predicted_classes = [1 if p > 0.5 else 0 for p in predictions]
#     accuracy = accuracy_score(true_labels, predicted_classes)
#     auc_score = roc_auc_score(true_labels, predictions)
#     print(f'Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}')

# if __name__ == '__main__':
#     from data_preprocessing import get_data_loaders
#     from torchvision import models
#     from torchvision.models import ResNet18_Weights
#     # Set device to CUDA if available, otherwise use CPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Initialize the feature extractor
#     feature_extractor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#     feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])  # Remove the last layers
#     # Initialize the lightweight network (same architecture as used during training)
#     input_channels = 512  # The output size of ResNet18's last convolutional layer
#     hidden_dim = 256
#     num_layers = 6
#     model = LightweightNetwork(input_channels, hidden_dim, num_layers)
#     # Load the trained model state
#     model.load_state_dict(torch.load('models/fpt_plus_trained_model.pth', map_location=device))
#     model = model.to(device)  # Move to the appropriate device (CPU/GPU)
#     feature_extractor = feature_extractor.to(device)
#     # Load the test data
#     data_dir = 'data/'  # Update this to the actual path of your data
#     _, _, test_loader = get_data_loaders(data_dir)
#     # Evaluate the model
#     evaluate_model(model, feature_extractor, test_loader)


# # # evaluation.py
# # import torch
# # from sklearn.metrics import accuracy_score, roc_auc_score
# # import torch.nn.functional as F

# # def evaluate_model(model, feature_extractor, test_loader):
# #     model.eval()
# #     feature_extractor.eval()
# #     predictions = []
# #     true_labels = []
# #     with torch.no_grad():
# #         for images, labels in test_loader:
# #             images = images.to('cuda')  # Move images to GPU if available
# #             labels = labels.to('cuda')  # Move labels to GPU if available
            
# #             features = feature_extractor(images)
# #             outputs = model(features)

# #             probs = F.softmax(outputs, dim=1)  # Get class probabilities
# #             predictions.extend(probs[:, 1].cpu().numpy())  # Assuming class 1 is the positive class
# #             true_labels.extend(labels.cpu().numpy())

# #     # Calculate accuracy and AUC
# #     predicted_classes = [1 if p > 0.5 else 0 for p in predictions]
# #     accuracy = accuracy_score(true_labels, predicted_classes)
# #     auc_score = roc_auc_score(true_labels, predictions)

# #     print(f'Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}')

# # if __name__ == '__main__':
# #     from data_preprocessing import get_data_loaders
# #     from torchvision import models
# #     from torchvision.models import ResNet18_Weights
# #     # Load the trained models
# #     feature_extractor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# #     feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])  # Remove the last layers
# #     model = torch.nn.Linear(512, 2)  # Assuming binary classification
# #     # Load the saved model state
# #     model.load_state_dict(torch.load('models/fpt_plus_trained_model.pth'))
# #     model = model.to('cuda')  # Move to GPU if available
# #     feature_extractor = feature_extractor.to('cuda')
# #     # Load test data
# #     data_dir = '/data'
# #     _, _, test_loader = get_data_loaders(data_dir)
# #     # Evaluate the model
# #     evaluate_model(model, feature_extractor, test_loader)
