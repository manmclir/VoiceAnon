import numpy as np
import torch, torchaudio
import requests
from vocos import Vocos

def create_wav():
    
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    checkpoint_path = 'vocos_checkpoint_epoch=639_step=1886720_val_loss=2.4528.ckpt'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    model_state_dict = vocos.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

# Load the filtered state_dict
    vocos.load_state_dict(filtered_state_dict, strict=False)
    data_lists = ['libri_dev_enrolls', 'libri_dev_trials_m', 'libri_dev_trials_f', 'libri_test_enrolls', 'libri_test_trials_m', 'libri_test_trials_f', 'IEMOCAP_dev', 'IEMOCAP_test']
    #data_lists = ['libri_dev_enrolls']
    #data_lists = ['libri_dev_trials_m', 'libri_dev_trials_f', 'libri_test_enrolls', 'libri_test_trials_m', 'libri_test_trials_f', 'IEMOCAP_dev', 'IEMOCAP_test', 'train-clean-360']
    data_lists = ['train-clean-360']
    for entry in data_lists:
        with open('data/' + entry + '/wav'+ '.scp', 'r') as scp_files:
            
            for line in scp_files:
                tokens = line.strip().split()

                wav_path = tokens[5]#the index is 5 because at this point I was anonymizing train-clean-360 for the others the index should be 1
                print(tokens)
                source, sr = torchaudio.load(wav_path)
                source = torchaudio.functional.resample(source, sr, 24000)
                if source.size(0) > 1:  # mix to mono
                    source = source.mean(dim=0, keepdim=True)
                target = vocos(source)
                target = torchaudio.functional.resample(target, 24000, 16000)

                print(line)
                wav_path_entry =  'data/' + entry + '_voco/wav/' 
                
                file_name = tokens[0] + '.wav'
                #target = torch.tensor(target.squeeze())
                #target = target.cpu()
                #target = target.unsqueeze(0)
                torchaudio.save(wav_path_entry + file_name, target, 16000)
                
                

if __name__ == "__main__":
    create_wav()

