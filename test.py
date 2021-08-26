from pyDiarization import speakerDiarization
import numpy
import sys
import os
from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt


if __name__ == "__main__":
    prev_mt_feats_norm = numpy.array([])
    prev_mt_feats_norm_or = numpy.array([])
    prev_mt_feats = numpy.array([])
    prev_cls = numpy.array([])
    class_names = ""
    i=0

    dir = "chunks"
    mt_size = float(2.0)
    mt_step = float(0.2)


    while os.path.isfile("{}/chunk{}.wav".format(dir, i)):
        file = "{}/chunk{}.wav".format(dir, i)
        cls, curr_mt_feats, class_names, centers = speakerDiarization(file, 0, mt_size, mt_step, .05, 0, False, prev_mt_feats)
        prev_mt_feats = curr_mt_feats
        i += 1

    print(cls)
    [fs, x] = audioBasicIO.read_audio_file("test.wav")
    duration = len(x) / fs
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_yticks(numpy.array(range(len(class_names))))
    ax1.axis((0, duration, -1, len(class_names)))
    ax1.set_yticklabels(class_names)
    ax1.plot(numpy.array(range(len(cls)))*mt_step+mt_step/mt_size, cls)
    plt.show()

    #print(speakerDiarization("data/diarizationExample.wav",0,1.0,.2,.05,0,False))
    #print(aS.speakerDiarization("{}.wav".format(dir) , 0 , 1.0, .2, .05, 0, True))

