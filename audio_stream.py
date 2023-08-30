import pyaudio

class Audio:
    def __init__(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 2
        WAVE_OUTPUT_FILENAME = "recordedFile.wav"
        self.audio = pyaudio.PyAudio()

        # id si spdif suka berubah, pastikan cek di 
        # python -m sounddevice
        # output = speakers - usb pnp audio device
        # input = analog input - built in audio device
        # speaker eksternal dicabut
        # device_index = 0
        # spidf
        device_index = 26
        # menggunakan headset bapak pulse, input gak bisa
        # device_index = 31 
        # sidf
        # device_index = 29
        # pulse
        # device_index = 29

        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,input_device_index = device_index,
                frames_per_buffer=CHUNK)
        
    def get_stream(self):
        # self.stream = audio.open(format=FORMAT, channels=CHANNELS,
        #         rate=RATE, input=True,input_device_index = device_index,
        #         frames_per_buffer=CHUNK)
        return self.stream

    def get_audio(self):
        # return self.audio
        return self.audio

