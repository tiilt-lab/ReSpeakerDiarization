from pyannote.core import Segment
from pyannote.audio.features import RawAudio
import torch
from pyAudioAnalysis import audioBasicIO
import numpy as np

import soundfile as sf
import csv


def speakerDiarization(x, fs, filename, chunkNum, chunkLength=60):
    '''
        ARGUMENTS:
            - x             the audio signal as a byte array
            - fs            the sampling rate of the audio passed in
            - filename      the name of the WAV file to store the audio
            - chunkNum      the number of times the audio has already been processed
            - chunkLength   the interval in seconds between each time the audio is processed
    '''

    if chunkNum == 0:
        sf.write(filename, x, fs)

    [signal, sampling_rate] = sf.read(filename)

    if chunkNum != 0:
        signal = np.concatenate((signal, x))
        sf.write(filename, signal, sampling_rate)
    chunkNum = 1 if chunkNum == 0 else chunkNum

    signal = audioBasicIO.stereo_to_mono(signal)
    duration = len(audioBasicIO.stereo_to_mono(signal)) / sampling_rate

    # no processing occurs if interval is not reached, audio is stored as wav file
    if duration < chunkLength * chunkNum:
        print("Need more audio first")
        return -1, -1, chunkNum

    chunkedFilename = f'{filename[:len(filename) - 4]}_byChunk.wav'

    waveform = RawAudio(sample_rate=sampling_rate)\
        .crop({'audio': filename}, Segment(0, chunkNum * chunkLength))
    sf.write(chunkedFilename, waveform, sampling_rate)

    pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')  # can use 'dia' or 'dia_ami'
    print('diarization begins')
    diarization = pipeline({'audio': chunkedFilename})

    speakers = set()
    timings = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)   # speakers represented as 'A','B',...,'Z','AA','AB',...
        timings.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        })

    return len(speakers), timings, (chunkNum + 1)  # refer to csv file for return values


# testing code
if __name__ == "__main__":

    chunkNum = 0
    for i in range(7):
        print(f'--ROUND {i}--')
        f_name = f'chunks/chunk{i}.wav'
        [np_array, rate] = sf.read(f_name)
        speakers, timings, chunkNum = speakerDiarization(np_array, 44100, 'testAudio.wav', chunkNum)

        print(f'chunk no.: {chunkNum}\nspeakers: {speakers}\n{timings}\n')

        if speakers != -1:
            with open('testAudio.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['speaker', 'start', 'end'])
                writer.writeheader()
                for data in timings:
                    writer.writerow(data)
