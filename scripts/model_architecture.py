import torch
import torch.nn as nn


class FineGrainedPrompt(nn.Module):
    def __init__(self, num_prompts, prompt_dim):
        super(FineGrainedPrompt, self).__init__()
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, prompt_dim))

    def forward(self, batch_size):
        # Repeat prompts for each sample in the batch
        return self.prompt_embeddings.expand(batch_size, -1, -1)


class FusionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FusionModule, self).__init__()
        self.fusion_layer = nn.Linear(input_dim * 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, vit_features, prompt_embeddings):
        # Flatten prompt_embeddings to 2D by averaging over the sequence dimension
        prompt_embeddings = prompt_embeddings.mean(dim=1)  # Shape becomes (batch_size, prompt_dim)
        
        # Concatenate features along the last dimension
        fused_features = torch.cat((vit_features, prompt_embeddings), dim=-1)
        return self.relu(self.fusion_layer(fused_features))


class LightweightNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_layers):
        super(LightweightNetwork, self).__init__()
        self.input_dim = input_channels

        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 2)  # Assuming binary classification

    def forward(self, x):
        # No pooling since we are dealing with features directly
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)  
        return x

