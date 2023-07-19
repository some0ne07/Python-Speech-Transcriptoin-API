from pydub import AudioSegment

def convert_mp4_to_wav(mp4_file):

    wav_file = '/content/output.wav'

    # Load the MP4 file
    audio = AudioSegment.from_file(mp4_file)

    # Set channels to mono
    audio = audio.set_channels(1)

    # Export the audio to a WAV file
    audio.export(wav_file, format='wav')

    return wav_file