import pydub
import numpy as np


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2, 0)
    aa = max(0,xx - a - h)
    b = max(0,(yy - w) // 2)
    bb = max(yy - b - w, 0)

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def load_audio(path, duration):
    audio = pydub.AudioSegment.silent(duration=duration)
    audio = audio.overlay(
        pydub.AudioSegment.from_file(path).set_frame_rate(22050).set_sample_width(2).set_channels(1))[0:duration]
    raw = (np.frombuffer(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  # convert to float
    return raw


def generate_features(y_cut):
    max_size = 1000  # my max audio file feature width
    stft = padding(np.abs(librosa.stft(y_cut, n_fft=255, hop_length=512)), 128, max_size)
    MFCCs = padding(librosa.feature.mfcc(y_cut, n_fft=n_fft, hop_length=hop_length, n_mfcc=128), 128, max_size)
    spec_centroid = librosa.feature.spectral_centroid(y=y_cut, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y_cut, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y_cut, sr=sr)

    # Now the padding part
    image = np.array([padding(normalize(spec_bw), 1, max_size)]).reshape(1, max_size)
    image = np.append(image, padding(normalize(spec_centroid), 1, max_size), axis=0)

    # repeat the padded spec_bw,spec_centroid and chroma stft until they are stft and MFCC-sized
    for i in range(0, 9):
        image = np.append(image, padding(normalize(spec_bw), 1, max_size), axis=0)
        image = np.append(image, padding(normalize(spec_centroid), 1, max_size), axis=0)
        image = np.append(image, padding(normalize(chroma_stft), 12, max_size), axis=0)
    image = np.dstack((image, np.abs(stft)))
    image = np.dstack((image, MFCCs))

    return image


def log_mel_features(waveform, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    log_mel_spectrogram = log_mel_spectrogram.astype(np.float32)
    return log_mel_spectrogram


def get_label(filename):
    label = {'engine_idling': 0, 'gun_shot': 1, 'siren': 2, 'dog': 3, 'insects': 4,
             'rain': 5, 'crickets': 6, 'breathing': 7, 'footsteps': 8, 'engine': 9,
             'car_horn': 10, 'train': 11, 'airplane': 12, 'fireworks': 13}
    return label[filename]


if __name__ == '__main__':
    pass
