from model import VEModel
from dataset import get_data
import torch
import torch.nn as nn
import torch.optim as optim
from training import Train
from evaluate import Evaluate

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,val_loader,label_mapping=get_data("/kaggle/input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data")
    model = VEModel(num_classes=len(label_mapping))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = Train(train_loader,model, criterion, optimizer, device)
        val_loss, val_accuracy = Evaluate(val_loader,model, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

   

if __name__=="__main__":
    run()