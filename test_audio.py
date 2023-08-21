import pyaudio
import wave
import librosa
import tensorflow as tf
import numpy as np
import pyaudio
import wave
import os
from ctypes import *
from contextlib import contextmanager

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "recordedFile.wav"
device_index = 0
os.environ['KMP_DUPLICATE_LIB_OK']='True'
SAVED_MODEL_PATH = "model_augment_senin.h5"
SAMPLES_TO_CONSIDER = 22050

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,input_device_index = device_index,
        frames_per_buffer=CHUNK)

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

class _Keyword_Spotting_Service:
    model = None
    _mapping = [
        "Kanan",
        "Kiri",
        "Lanjutkan",
        "Stop",
        "Tidak",
        "Ya"
    ]
    _instance = None

    def predict(self, file_path):
        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        print(predictions)
        predicted_index = np.argmax(predictions)
        pred_value = predictions[0][predicted_index]
        if pred_value > 0.8 :
            print(pred_value)
            predicted_keyword = self._mapping[predicted_index]
        else:
            predicted_keyword = "Ulangi"
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(y=signal,sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            #MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance

class audio_port:
    def __init__(self) -> None:
        self.start()

    def start(self):
        Recordframes = []
        print ("recording started")

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK,exception_on_overflow=False)
            Recordframes.append(data)

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()
        keyword1 = kss.predict(WAVE_OUTPUT_FILENAME)
        print(f"PREDICT KEYWORDS: {keyword1}")
      
if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    record = audio_port()
    
    record()
    
    # stream.stop_stream()
    # stream.close()
    # audio.terminate()   
    # print ("recording stopped")





