import sys
import os
import pdb
import pandas as pd
import numpy as np
import pydub
import librosa
from sklearn.preprocessing import normalize
import torchaudio.transforms as T
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


# from helper import *

def padding(array, target_shape):
    result = np.zeros(target_shape)
    slices = tuple(slice(0, min(array.shape[dim], target_shape[dim])) for dim in range(len(array.shape)))
    result[slices] = array[slices]
    return result


def log_mel_features(waveform, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    log_mel_spectrogram = log_mel_spectrogram.astype(np.float32)
    return log_mel_spectrogram


def get_label(filename):
    label = {'dog': 0, 'insects': 1, 'rain': 2, 'crickets': 3, 'breathing': 4, 'footsteps': 5, 'engine': 6,
             'car_horn': 7, 'train': 8, 'airplane': 9, 'fireworks': 10, 'siren': 11}
    # label = {'engine_idling': 0, 'gun_shot': 1, 'siren': 2, 'dog': 3, 'insects': 4,
    #          'rain': 5, 'crickets': 6, 'breathing': 7, 'footsteps': 8, 'engine': 9,
    #          'car_horn': 10, 'train': 11, 'airplane': 12, 'fireworks': 13}
    return label[filename]


class AudioDataset(Dataset):
    def __init__(self, audio_dir, target_length=173, config=None):
        self.audio_dir = audio_dir
        self.audio_files = os.listdir(audio_dir + '/ESC-50-relevant/audio')
        # (os.listdir(audio_dir + '/UrbanSound8k-relevant/audio')
        #                             + os.listdir(audio_dir + '/ESC-50-relevant/audio'))
        self.metadata = pd.read_csv(self.audio_dir + '/ESC-50-relevant/metadata/relevant_esc50.csv')
        # pd.read_csv(self.audio_dir + '/merged_relevant.csv')
        self.target_length = target_length
        self.config = config

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        filename = self.metadata['filename'][idx].split('/')[-1]
        audio_path = self.metadata['filename'][idx]
        # waveform = load_audio(audio_path, 4000)
        # pdb.set_trace()
        # load
        waveform, sr = librosa.load(audio_path, sr=self.config['sample_rate'], mono=True, duration=self.config['duration'])
        # pdb.set_trace()

        # feature generation
        # features = generate_features(waveform)
        features = log_mel_features(waveform, sr, n_fft=self.config['n_fft'], hop_length=self.config['hop_length'],
                                    n_mels=self.config['n_mels'])
        # pdb.set_trace()

        # Padding
        if features.shape[1] < self.target_length:
            features = padding(features, (128, self.config['target_length']))
        else:
            features = features[:, :self.config['target_length']]

        category = get_label(self.metadata['category_name'][idx])

        return features, category


def main():
    audio_dir = '/home/lucas/ESC/data'

    with open('/home/lucas/ESC/cnn/config.json', 'r') as f:
        config = json.load(f)

    dataset = AudioDataset(audio_dir=audio_dir, config=config)

    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for i, (features, category) in enumerate(data_loader):
        print(f"Batch {i + 1}")
        print(features.shape, category)


if __name__ == "__main__":
    main()
