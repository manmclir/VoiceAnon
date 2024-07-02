import numpy as np
import torch, torchaudio
import requests

def create_wav_test():
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()
    acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft", trust_repo=True).cuda()
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft", trust_repo=True).cuda()
    data_lists = ['libri_dev_enrolls', 'libri_dev_trials_m', 'libri_dev_trials_f', 'libri_test_enrolls', 'libri_test_trials_m', 'libri_test_trials_f', 'IEMOCAP_dev', 'IEMOCAP_test', 'train-clean-360']
    for entry in data_lists:
        print(entry)
        with open('data/' + entry + '/wav'+ '.scp', 'r') as scp_files:
            
            for line in scp_files:
                tokens = line.strip().split()
                wav_path = tokens[1]#the index should be changed to 5 when anonymizing trainclean-360
                source, sr = torchaudio.load(wav_path)
                source = torchaudio.functional.resample(source, sr, 16000)
                source = source.unsqueeze(0).cuda()
                with torch.inference_mode():
                    # Extract speech units
                    units = hubert.units(source)
                    # Generate target spectrogram
                    mel = acoustic.generate(units).transpose(1, 2)
                    # Generate audio waveform
                    target = hifigan(mel)
                wav_path_entry =  'data/' + entry + '_soft/wav/' 
                file_name = tokens[0] + '.wav'
    
    
if __name__ == "__main__":
    create_wav_test()
