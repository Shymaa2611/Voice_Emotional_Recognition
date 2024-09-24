import torch
import os 

def save_checkpoint(checkpoint_dir,model,optimizer,num_epochs,val_loss,val_accuracy):
    os.makedirs(checkpoint_dir,exist_ok=True)
    checkpoint= os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_epochs': num_epochs,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, checkpoint)

    print(f"Final model checkpoint saved: {checkpoint}")


def load_checkpoint(checkpoint_dir, model, optimizer):
    checkpoint= os.path.join(checkpoint_dir, "checkpoint.pt")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    num_epochs = checkpoint['num_epochs']
    val_loss = checkpoint['val_loss']
    val_accuracy = checkpoint['val_accuracy']
    
    print(f"Checkpoint loaded from: {checkpoint}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return num_epochs, val_loss, val_accuracy
