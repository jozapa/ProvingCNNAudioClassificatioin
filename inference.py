import  torch
from cnn import  CNNNetwork
from train import ANNOTATIONS_FILE,SAMPLE_RATE,AUDIO_DIR,NUM_SAMPLES
import torchaudio
from dataseturban import UrbanSoundDataset
class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]
def predict(model,input,target,class_mapping):
    model.eval()
    with  torch.no_grad():
        predictions = model(input)
        #Tensor(1,10)  ->  [[0.1,0.001,...,0.6]] we want to take the maxium value because that will be our predictions                                                         
        predicted_index  =  predictions[0].argmax(0) #predictions 0 get the first arugment of predictions, in this case input,argmax get the biggest arg
        predicted =  class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted,expected    

if __name__=="__main__":
    #load back the model                             
    cnn =  CNNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)
    
    #load dataset
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
                            "cpu")
    #get a sample from dataset for inference
    input,target = usd[0][0],usd[0][1]  #[batch_size,num_channels,fr,time]
    input.unsqueeze_(0)
    #make an inference
    predicted, expected = predict(cnn,input,target,
                                  class_mapping)
    print(f"Predicted:'{predicted}', expected: '{expected}'")