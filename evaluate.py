import torch

def Evaluate(dataloader,model,criterion,optimizer,device):
    model.eval()
    total_loss=0
    correct=0
    with torch.no_grad():
     for batch in dataloader:
        input_features,labels=batch
        input_features,labels=input_features.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(input_features)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy