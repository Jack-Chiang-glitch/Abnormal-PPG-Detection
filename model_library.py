import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
import torch
import torch.nn.functional as F
import os
import scipy.fft


class ComplexClassifierNet(nn.Module):
    def __init__(self, input_dim=20, num_classes=2, dropout_rate=0.0):
        super(ComplexClassifierNet, self).__init__()
        
        # 批次正規化層的參數
        self.bn_momentum = 0.1
        
        # 主要分類器架構
        self.classifier = nn.Sequential(
            # 第一個區塊
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, momentum=self.bn_momentum),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第二個區塊
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=self.bn_momentum),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第三個區塊
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=self.bn_momentum),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第四個區塊
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=self.bn_momentum),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第五個區塊
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=self.bn_momentum),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第六個區塊
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=self.bn_momentum),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 輸出層
            nn.Linear(64, num_classes)
        )
        
        # 殘差連接的額外層
        self.residual1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, momentum=self.bn_momentum)
        )
        
        self.residual2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, momentum=self.bn_momentum)
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(normalized_shape=num_classes)
        
    def forward(self, x):
        # 主要路徑
        out = self.classifier[0:4](x)  # 第一個區塊
        
        # 第一個殘差連接
        res1 = self.residual1(x)
        out = out + res1
        
        out = self.classifier[4:8](out)  # 第二個區塊
        
        # 第二個殘差連接
        res2 = self.residual2(out)
        out = out + res2
        
        # 繼續通過剩餘的層
        out = self.classifier[8:](out)
        
        # 最終的 Layer Normalization
        out = self.layer_norm(out)
        
        return out


class ClassifierNet(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierNet, self).__init__()
        print("initi")
        self.classifier = nn.Sequential(
            # 第一層
            nn.Linear(20, 128),              # 20 -> 128
            nn.BatchNorm1d(128),             # 批次正規化
            nn.ReLU(),
            
            # 第二層
            nn.Linear(128, 256),             # 128 -> 256
            nn.BatchNorm1d(256),             # 批次正規化
            nn.ReLU(),
            
            # 第三層
            nn.Linear(256, 128),             # 256 -> 128
            nn.BatchNorm1d(128),             # 批次正規化
            nn.ReLU(),
            
            # 第四層
            nn.Linear(128, 64),              # 128 -> 64
            nn.BatchNorm1d(64),              # 批次正規化
            nn.ReLU(),
            
            # 第五層
            nn.Linear(64, 32),               # 64 -> 32
            nn.BatchNorm1d(32),              # 批次正規化
            nn.ReLU(),
            
            # 輸出層
            nn.Linear(32, num_classes)        # 32 -> num_classes
        )
                    
    def forward(self, x):
        return self.classifier(x)


# Define the CNN Model
class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        
        # Layer Definitions based on the architecture table
        
        # Conv1 Layer
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Conv2 Layer
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, stride=1)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        
        # Conv3 Layer
        self.dropout3 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=8, stride=1)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm1d(16)
        #self.avg_pool = nn.AvgPool1d(kernel_size=2)
        ###

        # Conv4 layer
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=4, stride=1)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        
        # Feature Dense Layer
        #self.dropout4 = nn.Dropout(p=0.1)
        self.feature_dense1 = nn.Linear(20, 4)
        
        
        # Concatenation Layer is handled later
        
        # Output Layer
        self.output_dense1 = nn.Linear(52, num_classes)
        #self.output_dense2 = nn.Linear(12, 6)
        #self.output_dense3 = nn.Linear(6, num_classes)  # Adjust according to the number of classes
        self.print = False
        
    def forward(self, x, feature):
        # First conv block
        x = self.dropout1(x)
        x = self.conv1(x);  print("Conv1 : " + str(x.shape)) if self.print else None
        x = F.relu(x)
        x = self.pool1(x);  print("Pool1 : " + str(x.shape)) if self.print else None
        x = self.bn1(x)
        # Second conv block
        x = self.dropout2(x)
        x = self.conv2(x);  print("Conv2 : " + str(x.shape)) if self.print else None
        x = F.relu(x)
        x = self.pool2(x);  print("Pool2 : " + str(x.shape)) if self.print else None
        x = self.bn2(x) 
        # Third conv block
        x = self.dropout3(x)
        x = self.conv3(x);  print("Conv3 : " + str(x.shape)) if self.print else None
        x = F.relu(x)
        x = self.pool3(x);  print("Pool3 : " + str(x.shape)) if self.print else None
        x = self.bn3(x)
        # Fourth conv block
        x = self.conv4(x);  print("Conv4 : " + str(x.shape)) if self.print else None
        x = F.relu(x)
        x = self.pool4(x);  print("Pool4 : " + str(x.shape)) if self.print else None
        
        
        # Flatten the output
        x = x.view(x.size(0), -1);  print("Flatten : " + str(x.shape)) if self.print else None

        
        # Feature dense layer
        feature = self.feature_dense1(feature)
        feature = F.relu(feature)
        feature = feature.view(x.size(0), -1); print("Feature : " + str(feature.shape)) if self.print else None
        
        
        # Concatenate the CNN output with the feature dense layer output
        combined = torch.cat((x, feature), dim=1); print("Combined : " + str(combined.shape)) if self.print else None
        
        # Final output layer
        out = self.output_dense1(combined)
        return out


class Autoencoder_Attention(nn.Module):
    def __init__(self, input_size=1, latent_size=20, hidden_size=128):
        super(Autoencoder_Attention, self).__init__()

        # Attention Encoder: 將序列壓縮到 latent_size 維
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Linear layers to produce Query, Key, and Value
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

        # 線性層將注意力輸出壓縮到 latent_size
        self.fc_latent = nn.Linear(hidden_size, latent_size)

        # Decoder: 將 latent_size 解碼回原序列
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 400),
            nn.ReLU(),
            nn.Linear(400, 400)
        )

    def forward(self, x):
        # x 的形狀: (batch_size, seq_len, input_size)

        # 計算 Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)    # (batch_size, seq_len, hidden_size)
        V = self.value(x)  # (batch_size, seq_len, hidden_size)

        # 計算注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)  # (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)                              # (batch_size, seq_len, seq_len)

        # 應用注意力權重到 Value
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, hidden_size)

        # 將注意力輸出取平均，壓縮到 latent_size 維
        attention_pooled = attention_output.mean(dim=1)        # (batch_size, hidden_size)
        encoded = self.fc_latent(attention_pooled)             # (batch_size, latent_size)

        # Decoder 解碼回原序列
        decoded = self.decoder(encoded)                        # (batch_size, input_size)

        return encoded, decoded


class Autoencoder_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, latent_size=20, num_layers=2):
        super(Autoencoder_LSTM, self).__init__()
        
        # LSTM Encoder: 將 400 維輸入壓縮到 20 維
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_latent = nn.Linear(hidden_size, latent_size)
        
        # 原本的 Decoder: 將 20 維解碼回 400 維
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 400)
        )
        
    def forward(self, x):
        # LSTM Encoder
        #print(x.shape)
        # x 的形狀: (batch_size, sequence_length=1, input_size=400)
        _, (H_n, _) = self.encoder(x)  # h_n: (num_layers, batch_size, hidden_size)
        #print('H_n')
        #print(H_n.shape)
        
        # 取最後一層的隱藏狀態作為編碼表示
        encoded = H_n[-1]  # (batch_size, hidden_size)
        #print(encoded.shape)
        # 經過全連接層壓縮到 latent_size
        encoded = self.fc_latent(encoded)  # (batch_size, latent_size)
        
        # Decoder
        decoded = self.decoder(encoded)  # (batch_size, input_size)
        
        return encoded, decoded
        


# 自編碼器類
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: 將 400 維壓縮到 20 維
        self.encoder = nn.Sequential(
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 78),
            nn.ReLU(),
            nn.Linear(78, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 30)
        )
        # Decoder: 將 20 維解碼回 400 維
        self.decoder = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 78),
            nn.ReLU(),
            nn.Linear(78, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 500),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



class Autoencoder_Batchnormal(nn.Module):
    def __init__(self):
        super(Autoencoder_Batchnormal, self).__init__()
        # Encoder: 將 500 維壓縮到 30 維
        self.encoder = nn.Sequential(
            nn.Linear(500, 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, 78),
            nn.BatchNorm1d(78),   # Batch Normalization
            nn.ReLU(),
            nn.Linear(78, 64),
            nn.BatchNorm1d(64),   # Batch Normalization
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),   # Batch Normalization
            nn.ReLU(),
            nn.Linear(32, 30)
        )
        
        # Decoder: 將 30 維解碼回 500 維
        self.decoder = nn.Sequential(
            nn.Linear(30, 32),
            nn.BatchNorm1d(32),   # Batch Normalization
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),   # Batch Normalization
            nn.ReLU(),
            nn.Linear(64, 78),
            nn.BatchNorm1d(78),   # Batch Normalization
            nn.ReLU(),
            nn.Linear(78, 128),
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(256, 500)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



        
        


        
    
        
        

