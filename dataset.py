from transformers import WhisperProcessor
import torch
from torch.utils.data import Dataset,DataLoader
import os
import torchaudio


class VoiceEmotionDataset(Dataset):
    def __init__(self,root_dir,processor,label_mapping):
        self.root_dir=root_dir
        self.processor=processor
        self.label_mapping = label_mapping
        self.audio_paths = []
        self.labels = []
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for audio_file in os.listdir(label_dir):
                    if audio_file.endswith((".wav")):
                        self.audio_paths.append(os.path.join(label_dir, audio_file))
                        self.labels.append(self.label_mapping[label])

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        label = self.labels[index]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        inputs = self.processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000)
        return inputs.input_features.squeeze(0), torch.tensor(label)

def create_label_mapping(root_dir):
    labels = {label: idx for idx, label in enumerate(os.listdir(root_dir))}
    return labels

def get_data(root_dir):
    label_mapping = create_label_mapping(root_dir)
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    dataset = VoiceEmotionDataset(root_dir, processor, label_mapping)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return train_loader,val_loader,label_mapping







