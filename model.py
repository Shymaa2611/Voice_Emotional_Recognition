from transformers import WhisperModel
import torch
import torch.nn as nn


class VEModel(nn.Module):
    def __init__(self,num_classes):
        super(VEModel, self).__init__()
        self.model=WhisperModel.from_pretrained("openai/whisper-base")
        self.classifier=nn.Linear(self.model.config.d_model,num_classes)

    def forward(self,audio_features):
        out=self.model.encoder(audio_features)
        embeddings = out.last_hidden_state.mean(dim=1) 
        logits=self.classifier(embeddings)
        return logits
    
    
