from pyAudioAnalysis import audioSegmentation as aS

print(aS.speakerDiarization("data/diarizationExample.wav",0,2.0,.2,.05,35,True))
