import torch
import torch.nn as nn

class SpellingDifficultyModel(nn.Module):
    def __init__(self, vocab_size=28, char_embed_dim=64, num_handcrafted_features=50, num_user_features=9):
        super(SpellingDifficultyModel, self).__init__()
        
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(char_embed_dim, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(256, 512, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=4, padding=1),
            nn.ReLU(),
        )
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        cnn_output_dim = 512
        total_features = cnn_output_dim + num_handcrafted_features + num_user_features
        
        self.fc_layers = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, char_indices, handcrafted_features, user_features):
        char_embeds = self.char_embedding(char_indices)
        char_embeds = char_embeds.permute(0, 2, 1)
        
        x = self.conv_layers(char_embeds)
            
        cnn_features = self.pool(x).squeeze(-1)
        
        combined = torch.cat([cnn_features, handcrafted_features, user_features], dim=-1)
        difficulty = self.fc_layers(combined)
        return difficulty
