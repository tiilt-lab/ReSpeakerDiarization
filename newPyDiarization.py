from pyannote.core import Segment
from pyannote.audio.features import RawAudio
import torch
from pyAudioAnalysis import audioBasicIO
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import read
import soundfile as sf
import csv


def speakerDiarization(x, sample_width, sample_rate, n_channels, path, chunk_num, chunk_length=60):
    '''
        ARGUMENTS:
            - data          the audio signal as a byte array (np ndarray)
            - sample_width  the sample_width (bytes) of the audio signal
            - sample_rate   the sampling rate (Hz) of the audio signal
            - n_channels     the number of channels of the audio signal
            - path          the path of the WAV file to store the audio
            - chunk_num      the number of times the audio has already been processed;
                            initiate at 0, pass in returned value afterwards
            - chunk_length   the interval in seconds between each time the audio is processed
    '''

    if chunk_num == 0:
        new = AudioSegment(data=x.tobytes(),
                           sample_width=sample_width,
                           frame_rate=sample_rate,
                           channels=n_channels)
        new.export(path, format='wav')

    audio = AudioSegment.from_file(path, format='wav')

    if chunk_num != 0:
        new = AudioSegment(data=x.tobytes(),
                           sample_width=audio.sample_width,
                           frame_rate=audio.frame_rate,
                           channels=audio.channels)
        # append new audio to old audio stored on file
        audio = audio + new
        audio.export(path, format='wav')

    chunk_num = 1 if chunk_num == 0 else chunk_num

    monoAudio = audio.set_channels(1)
    duration = audio.duration_seconds

    # no processing occurs if interval is not reached, audio is stored as wav file
    if duration < chunk_length * chunk_num:
        print("Need more audio first")
        return -1, -1, chunk_num

    monoPath = f'mono_{path}'
    audio.export(monoPath, format='wav')

    pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')  # can use 'dia' or 'dia_ami'
    print('diarization begins')
    diarization = pipeline(dict(audio=monoPath))

    speakers = set()
    timings = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)   # speakers represented as 'A','B',...,'Z','AA','AB',...
        timings.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        })

    return len(speakers), timings, (chunk_num + 1)  # refer to csv file for return values


# testing code
if __name__ == "__main__":

    chunkNum = 0
    for i in range(7):
        print(f'--ROUND {i}--')
        f_name = f'chunks/chunk{i}.wav'
        rate, signal = read(f_name)
        speakers, timings, chunkNum = speakerDiarization(signal, 2, 44100, 2, 'testAudio.wav', chunkNum)

        print(f'chunk no.: {chunkNum}\nspeakers: {speakers}\n{timings}\n')

        if speakers != -1:
            with open('testAudio.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['speaker', 'start', 'end'])
                writer.writeheader()
                for data in timings:
                    writer.writerow(data)
