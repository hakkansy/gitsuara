import librosa
import tensorflow as tf
import numpy as np
import pyaudio
import wave
import os
# import keyboard
# from pynput import keyboard

from ctypes import *
from contextlib import contextmanager
import pyaudio
import time

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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# SAVED_MODEL_PATH = "modelbaru3.h5"
SAVED_MODEL_PATH = "model_augment_senin.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.

    :param model: Trained model
    
    """

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
        """

        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

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
        """Extract MFCCs from audio file.

        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples

        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

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
    """Factory function for Keyword_Spotting_Service class.

    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance




if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    #kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    #assert kss is kss1

    # make a prediction
    # keyword1 = kss.predict("test/kanan.wav")
    # keyword2 = kss.predict("test/lanjutkan.wav")
    # # keyword3 = kss.predict("test/tidak.wav")
    # keyword3 = kss.predict("test/tidak1.wav")
    # # keyword4 = kss.predict("test/ya.wav")
    # keyword4 = kss.predict("test/out_lanjutkan.wav")
    # keyword5 = kss.predict("test/stop.wav")
    # keyword6 = kss.predict("test/out_kanan.wav")
    # #print(f"PREDICT KEYWORDS: {keyword1},{keyword2}")
    # print(f"PREDICT KEYWORDS: {keyword1}")
    # print(f"PREDICT KEYWORDS: {keyword2}")
    # print(f"PREDICT KEYWORDS: {keyword3}")
    # print(f"PREDICT KEYWORDS: {keyword4}")
    # print(f"PREDICT KEYWORDS: {keyword5}")
    # print(f"PREDICT KEYWORDS: {keyword6}")


    #AUDIO INPUT
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "shi.wav"
# WAVE_OUTPUT_FILENAME = "recordedFile.wav"
index = 35

with noalsaerr():
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, input_device_index = index,
                    frames_per_buffer=CHUNK,
                    )

    
    while(1):
    #   print("recording")
        ui = input("tekan 1\n")
        if ui == "1":
            print ("recording started")
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print ("recording stopped")
 
            stream.stop_stream()
            stream.close()
            audio.terminate()

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            # spf = wave.open(WAVE_OUTPUT_FILENAME,'r')
            # add = "recordedFile.wav"
            keyword1 = kss.predict(WAVE_OUTPUT_FILENAME)
            # keyword2 = kss.predict("Ya78_3_pitch.wav")
            # keyword3 = kss.predict("Tidak156_4_rGain.wav")
            # keyword4 = kss.predict("Kiri35_2_rGain.wav")
            # keyword5 = kss.predict("kanan1.wav")
            # keyword6 = kss.predict("recordedFile.wav")
            
            print(f"PREDICT KEYWORDS: {keyword1}")
            # print(f"PREDICT KEYWORDS: {keyword2}")
            # print(f"PREDICT KEYWORDS: {keyword3}")
            # print(f"PREDICT KEYWORDS: {keyword4}")
            # print(f"PREDICT KEYWORDS: {keyword5}")
            # print(f"PREDICT KEYWORDS: {keyword6}")
            
            # frames = []

        # Extract Raw Audio from Wav File
            # signal = spf.readframes(-1)
            # signal = np.frombuffer(signal, dtype=np.int16)   
            # copy= signal.copy()
    
    # while(1):
    # #   print("recording")
    #     # if keyboard.is_pressed('space'):
    #         frames = []
    #         for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #             data = stream.read(CHUNK)
    #             frames.append(data)
    #         waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    #         waveFile.setnchannels(CHANNELS)
    #         waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    #         waveFile.setframerate(RATE)
    #         waveFile.writeframes(b''.join(frames))
    #         waveFile.close()
    #         spf = wave.open(WAVE_OUTPUT_FILENAME,'r')
    #         keyword1 = kss.predict(WAVE_OUTPUT_FILENAME)
    #         print(f"PREDICT KEYWORDS: {keyword1}")
    #         frames = []

    # #Extract Raw Audio from Wav File
    #         signal = spf.readframes(-1)
    #         signal = np.frombuffer(signal, dtype=np.int16)   
    #         copy= signal.copy()
        

    # listener = keyboard.Listener(on_press=predict_record)
    # listener.start()
    # listener.join() 