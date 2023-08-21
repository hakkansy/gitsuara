import time
from pygame import mixer

		
def run(path):
		#Play the music
				#Load audio file
		mixer.music.load(path)

		sound_duration = mixer.Sound(path)
		sound_duration = sound_duration.get_length()

		#Set preferred volume
		mixer.music.set_volume(0.2)

		# print("music started playing....")

		#Set preferred volume
	

		mixer.music.play()
		time.sleep(sound_duration)

if __name__ == "__main__":
	# play = Play_prompt()
	run('perintah.mpga')