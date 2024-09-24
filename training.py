
def Train(dataloader,model,criterion,optimizer,device):
    model.train()
    total_loss=0
    for batch in dataloader:
        input_features,labels=batch
        input_features,labels=input_features.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(input_features)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss / len(dataloader)