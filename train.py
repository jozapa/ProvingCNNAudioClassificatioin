from calendar import EPOCH
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataseturban import UrbanSoundDataset
import torchaudio
from cnn import CNNNetwork

BATCH_SIZE  =  128
EPOCHS = 2
LEARNING_RATE = .001

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs,  targets in data_loader:
        inputs,targets  = inputs.to(device), targets.to(device)

        #calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions,targets)
        #backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"Loss:{loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device,EPOCHS):
    for i in range(EPOCHS):
        print(f"Epoch{i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-----------")
    print("Training is done")

if __name__=="__main__":
    if  torch.cuda.is_available():
        device  = "cuda"
    else:
        device ="cpu"   
    print(f"Using device {device}")

    ANNOTATIONS_FILE  = "/home/josep/Documentos/CNNClassificator/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/home/josep/Documentos/CNNClassificator/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES =22050 #1 sec of audio

    mel_spectrogram  = torchaudio.transforms.MelSpectrogram(
       sample_rate=SAMPLE_RATE,
       n_fft=1024,
       hop_length=512,
       n_mels = 64 
    )

    
    usd  = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    #2 create data loader: allows data batches 
    train_data_loader = DataLoader(usd,batch_size=BATCH_SIZE)

    #3 build a model
     
    cnn = CNNNetwork().to(device)
     #instantiat loss function
    loss_fn= nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                   lr=LEARNING_RATE)  

    #train the model 
    train(cnn,train_data_loader,loss_fn,optimiser,device,EPOCHS)
    torch.save(cnn.state_dict(),"cnn.pth")
    print("Model Trained  and  sotored")
