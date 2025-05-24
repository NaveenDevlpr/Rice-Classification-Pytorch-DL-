import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device using", device)

data_df=pd.read_csv('riceClassification.csv')
print(data_df.head())