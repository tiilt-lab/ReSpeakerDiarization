from __future__ import print_function
import json
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback, AudioSource
import threading

# If service instance provides API key authentication
# service = SpeechToTextV1(
#     ## url is optional, and defaults to the URL below. Use the correct URL for your region.
#     url='https://stream.watsonplatform.net/speech-to-text/api',
#     iam_apikey='your_apikey')

service = SpeechToTextV1(
    username='8bffd6a4-e302-4326-a7bb-eadd9bd7bf44',
    password='Yp4wsCiPlBj8',
    url='https://stream.watsonplatform.net/speech-to-text/api')

models = service.list_models().get_result()
print(json.dumps(models, indent=2))

model = service.get_model('en-US_BroadbandModel').get_result()
print(json.dumps(model, indent=2))

with open(join(dirname(__file__), 'data/diarizationExample.wav'),
          'rb') as audio_file:

    data = service.recognize(audio=audio_file, content_type='audio/wav', timestamps=True, word_confidence=True, speaker_labels=True).get_result()
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)

'''
# Example using websockets
class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        print(transcript)

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_hypothesis(self, hypothesis):
        print(hypothesis)

    def on_data(self, data):
        print(data)

# ample using threads in a non-blocking way
mycallback = MyRecognizeCallback()
audio_file = open(join(dirname(__file__), 'data/diarizationExample.wav'), 'rb')
audio_source = AudioSource(audio_file)
recognize_thread = threading.Thread(
    target=service.recognize_using_websocket,
    args=(audio_source, "audio/wav; rate=44100, speaker_labels=true", mycallback))
recognize_thread.start()
'''
