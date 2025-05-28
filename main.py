import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device using", device)

data_df = pd.read_csv("riceClassification.csv")
print(data_df.head())

data_df = data_df.dropna()

original_df = data_df.copy()

for cols in data_df.columns:
    data_df[cols] = data_df[cols] / data_df[cols].max()
    
    
    
X=np.array(data_df.iloc[:,:-1])
Y=np.array(data_df.iloc[:,-1])

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
X_test,X_val,y_test,y_val=train_test_split(X_test,y_test,test_size=0.5,random_state=0)


class dataset(Dataset):
    def __init__(self,X,Y):
        self.X=torch.tensor(X,dtype=torch.float32).to(device)
        self.Y=torch.tensor(Y,dtype=torch.float32).to(device)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]
    
    
    
    
training_data=dataset(X_train,y_train)
validation_data=dataset(X_val,y_val)
test_data=dataset(X_test,y_test)


training_dataloader=DataLoader(training_data,batch_size=8,shuffle=True)
validation_dataloader=DataLoader(validation_data,batch_size=8,shuffle=True)   
test_dataloader=DataLoader(test_data,batch_size=8,shuffle=True)



class Model(nn.Module):
    def __init__(self,):
        super(Model,self).__init__()
        
        self.input_layer=nn.Linear(X.shape[1],20)
        self.linear=nn.Linear(20,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.input_layer(x)
        x=self.linear(x)
        x=self.sigmoid(x)
        return x
    
    
model=Model().to(device)



criterion=nn.BCELoss()
optimizer=Adam(model.parameters(),lr=1e-3)



total_loss_train_plot=[]
total_loss_valid_plot=[]
total_acc_train_plot=[]
total_acc_valid_plot=[]

epochs=10

for epoch in range(epochs):
    total_loss_train=0
    total_loss_valid=0
    total_acc_train=0
    total_acc_valid=0
    
    for data in training_dataloader:
        input,labels=data
        predictions=model(input).squeeze(1)
        batch_loss=criterion(predictions,labels)
        total_loss_train+=batch_loss.item()
        acc=(predictions.round()==labels).sum().item()
        total_acc_train+=acc
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    with torch.no_grad():
        for data in validation_dataloader:
            inputs,labels=data
            predictions=model(inputs).squeeze(1)
            batch_loss=criterion(predictions,labels)
            total_loss_valid+=batch_loss.item()
            acc=(predictions.round()==labels).sum().item()
            total_acc_valid+=acc
            
    total_loss_train_plot.append(round(total_loss_train/1000,4))
    total_loss_valid_plot.append(round(total_loss_valid/1000,4))
    
    total_acc_train_plot.append(round(total_acc_train/training_data.__len__() * 100,4))
    total_acc_valid_plot.append(round(total_acc_valid/validation_data.__len__() * 100,4))
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Training Loss: {round(total_loss_train/1000,4)}")
    print(f"Validation Loss: {round(total_loss_valid/1000,4)}")
    print(f"Training Accuracy: {round(total_acc_train/training_data.__len__() * 100,4)}")
    print(f"Validation Accuracy: {round(total_acc_valid/validation_data.__len__() * 100,4)}")
    print("=====================================")
    
    
    


with torch.no_grad():
    total_loss_test=0
    total_acc_test=0
    for data in test_dataloader:
        inputs,labels=data
        predictions=model(inputs).squeeze(1)
        batch_loss=criterion(predictions,labels)
        total_loss_test+=batch_loss.item()
        acc=(predictions.round()==labels).sum().item()
        total_acc_test+=acc
        
print("Accuracy: ",round(total_acc_test/test_data.__len__() * 100,4),"Loss: ",round(total_loss_test/1000,4))