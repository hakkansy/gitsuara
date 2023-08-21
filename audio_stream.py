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

        device_index = 0

        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,input_device_index = device_index,
                frames_per_buffer=CHUNK)
        
    def get_stream(self):
        return self.stream

    def get_audio(self):
        return self.audio

