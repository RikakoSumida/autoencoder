# coding: UTF-8
#https://tips-memo.com/python-ae
import os
import sys
import numpy as np
import librosa
import glob
import pickle

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset

from tools import EarlyStopping

def extract_mel(wav, sr, n_mels=64, hop_length=160, n_fft=512): #(timeframe, mel_dim) 
    audio, _ = librosa.load(wav, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels , hop_length=160, n_fft=512).T
    return mel

def data_list(path, sr, n_mels=64, hop_length=160, n_fft=512): #620
    wav_list = glob.glob(path)
    size = len(wav_list)
    data = np.ones((1, n_mels))
    count= 0
    for wavname in wav_list:
        component = extract_mel(wavname, sr=sr, n_mels=64, hop_length=160, n_fft=512)
        data = np.concatenate([data, component], axis=0)
        count += 1
        sys.stdout.write("\r%s" % "現在"+str(np.around((count/len(wav_list))*100 , 2))+"%完了")
        sys.stdout.flush()
    return data[1:], size

class ExpandDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        #print(self.transform)
        self.data = data
        #print(self.data)
        self.data_num = len(data)
        if self.data.ndim==2:
            self.pad_data_fr = data[:10][::-1] #0~9行目までを
            self.pad_data_bc = data[-10:][::-1]
            self.pad_data = np.concatenate([self.pad_data_fr, data, self.pad_data_bc], axis=0)
        elif self.data.ndim==3:
            self.pad_data_fr = data[:,:10,:][:,::-1,:]
            self.pad_data_bc = data[:,-10:,:][:,::-1,:]
            self.pad_data = np.concatenate([self.pad_data_fr, data, self.pad_data_bc], axis=1)    
    def __len__(self):
        return self.data_num
    def __getitem__(self, idx):
        if self.transform:
            if self.data.ndim==2:
                #__getitem__が呼ばれると,idxから20取ってきて，一次元配列に加工するƒ
                out_data = self.transform(self.pad_data)[0][idx:idx+20].flatten()
            elif self.data.ndim==3:
                index = int(random.uniform(0,self.data.shape[1]))
                out_data = self.transform(self.pad_data)[:,idx, index:index+20].flatten()
        else:
            print("transformを使用しテンソル化してください")
        return out_data

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.dense_enc1 = nn.Linear(1280, 1200)
        self.bn1 = nn.BatchNorm1d(1200)
        self.dense_enc2 = nn.Linear(1200, 1100)
        self.bn2 = nn.BatchNorm1d(1100)
        self.dense_enc3 = nn.Linear(1100,1024)
    
        self.dense_dec1 = nn.Linear(1024,1100)
        self.bn4 = nn.BatchNorm1d(1100)
        self.dense_dec2 = nn.Linear(1100, 1200)
        self.bn5 = nn.BatchNorm1d(1200)
        self.drop1 = nn.Dropout(p=0.2)
        self.dense_dec3 = nn.Linear(1200, 1280)

    def encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = self.bn1(x)
        x = F.relu(self.dense_enc2(x))
        x = self.bn2(x)
        x = self.dense_enc3(x)
        return x

    def decoder(self, x):
        x = F.relu(self.dense_dec1(x))
        x = self.bn4(x)
        x = F.relu(self.dense_dec2(x))
        x = self.bn5(x)
        x = self.drop1(x)
        x = self.dense_dec3(x)
        return x

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

def func_100(epoch):
    if epoch <= 100:
        return 1
    elif 100 < epoch:
        return -0.99*(1e-2)*(epoch) + 1.99

def train(model,train_loader,criterion,optimizer):
    model.train()
    train_losses = []
    for x in train_loader:
        x = x.to(device)
        model.zero_grad()
        y, z = model(x.float())
        
        loss = criterion(y.float(), x.float())
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
    return train_losses

def valid(model, valid_loader, criterion):
    model.eval()
    valid_losses = []
    for x in valid_loader:
        x = x.to(device)
        y, z = model(x.float())
        
        loss = criterion(y.float(), x.float())
        valid_losses.append(loss.item())
    return valid_losses

def train_model(model, batch_size, train_loader, valid_loader, patience, criterion, optimizer, num_epochs, PATH):
#def train_model(model, batch_size, train_loader, valid_loader, patience, criterion, optimizer, num_epochs):
        
    avg_train_losses = []
    avg_valid_losses = []
    
    early_stopping = EarlyStopping(patience = patience, verbose = True, path = PATH)
    
    for epoch in range(num_epochs):     
        train_losses = train(model,train_loader,criterion,optimizer)
        valid_losses = valid(model,valid_loader,criterion)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs))
        
        print_msg = ('[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}]' + 'train_loss: {train_loss:.5f}' + 'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        
        
        early_stopping(valid_loss,model)
        
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        
    #model.load_state_dict(torch.load('checkpoint.pt'))
    model.load_state_dict(torch.load(PATH))
    return model, avg_train_losses, avg_valid_losses


if __name__ == '__main__':
    DIR_NAME = '1214'
    os.makedirs(DIR_NAME, exist_ok=False)
    PATH = '{}/checkpoint.pt'.format(DIR_NAME)
    print('Start training Auto Encoder')

    # pathの設定
    train_wav_path= "./train/*.wav"
    valid_wav_path = "./valid/*.wav"

    # wavデータの一括読み込み
    # 今回はメル周波数スペクトログラムを15500Hz, 620次元取得して全データに関して縦に並べている
    print("---read_training_wav---")
    train_data_list, size_train = data_list(train_wav_path, sr=16000, hop_length=160, n_mels=64, n_fft=512)
    print("\n---read_valid_wav---")
    valid_data_list, size_test = data_list(valid_wav_path, sr=16000, hop_length=160, n_mels=64, n_fft=512)
    print("\n")

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('torch.cuda.is_available')

    # データセット
    transform = transforms.Compose([transforms.ToTensor()])
    print('loading train dataset')
    train_dataset = ExpandDataset(train_data_list, transform)
    train_batch_size = 100 #25
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    print('loading valid dataset')
    valid_dataset = ExpandDataset(valid_data_list, transform)
    valid_batch_size = 100 #25
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True)

    num_epochs = 10
    learning_rate = 1e-4

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=func_100)


    batch_size = 100 #256

    #train_loader, test_loader, valid_loader = create_datasets(batch_size)
    #train_loader, valid_loader = create_datasets(batch_size)

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 20

    #PATH
    print('Start training')
    model, train_loss, valid_loss = train_model(model, batch_size, train_loader, valid_loader, patience, criterion, optimizer, num_epochs, PATH)

    # 保存
    #torch.save(model.state_dict(), 'model.pt')
    torch.save(model.state_dict(), '{}/model.pt'.format(DIR_NAME))

    #with open('train_loss.pickle', mode='wb') as f:
        #pickle.dump(train_loss, f)
    with open('{}/train.pickle'.format(DIR_NAME), mode='wb') as f:
        pickle.dump(train_loss, f)

    #with open('valid_loss.pickle', mode='wb') as f:
        #pickle.dump(valid_loss, f)
    with open('{}/valid.pickle'.format(DIR_NAME), mode='wb') as f:
        pickle.dump(valid_loss, f)

