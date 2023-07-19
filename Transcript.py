import whisper
import datetime
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def get_embedding_model():
    return PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda"))

def get_whisper_model():
    return whisper.load_model('large')


def segment_embedding(segment: dict, path: str, duration: float, audio: Audio, embedding_model) -> np.array:
    """
    Generate an embedding for a segment of audio.

    Args:
        segment (dict): Segment information.
        path (str): Path to the audio file.
        duration (float): Duration of the audio in seconds.
        audio (Audio): Audio object for processing.

    Returns:
        np.array: Embedding for the segment.
    """
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def time(secs: float) -> datetime.timedelta:
    """
    Convert seconds to timedelta.

    Args:
        secs (float): Time in seconds.

    Returns:
        datetime.timedelta: Time in timedelta format.
    """
    return datetime.timedelta(seconds=round(secs))

def generate_transcript(segments: list) -> str:
    """
    Generate the transcript with speaker labels.

    Args:
        segments (list): List of audio segments.

    Returns:
        str: Generated transcript with speaker labels.
    """
    transcript = ""
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            transcript += f"\n{segment['speaker']} {str(time(segment['start']))}\n"
        transcript += segment["text"][1:] + " "
    return transcript

def generate_speaker_labeled_transcript(path: str, num_speakers: int, model, embedding_model) -> str:
    """
    Process audio file and generate a transcript with speaker labels.

    Args:
        path (str): Path to the audio file.
        num_speakers (int): Number of speakers to identify.

    Returns:
        str: Generated transcript with speaker labels.
    """

    # Transcribe the audio
    result = model.transcribe(path)
    segments = result["segments"]

    # Get audio duration
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment,path,duration,audio,embedding_model)
        
    # Replace NaN values with zeros
    embeddings = np.nan_to_num(embeddings)

    # Perform speaker clustering
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    # Assign speakers to segments
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    transcript = generate_transcript(segments)

    return transcript
