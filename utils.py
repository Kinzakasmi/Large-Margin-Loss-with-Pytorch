import torch
from torch.utils.data import TensorDataset, DataLoader

def create_loaders(X_train,X_test,y_train, y_test):
    #Creating train_loader
    x = torch.Tensor(X_train[:,:2]).float() # transform to torch tensor
    y = torch.Tensor(y_train).long()
    train_loader = DataLoader(TensorDataset(x,y), batch_size=4, shuffle=False, drop_last=True)

    #Creating test_loader
    x = torch.Tensor(X_test[:,:2]).float() # transform to torch tensor
    y = torch.Tensor(y_test).long()

    test_loader = DataLoader(TensorDataset(x,y),batch_size=8, shuffle=False, drop_last=True)

    return train_loader, test_loader