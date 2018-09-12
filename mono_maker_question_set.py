import subprocess
import os
import speech_recognition as sr
from google.cloud import speech_v1p1beta1 as speech
import io
import csv
import google.cloud
import json
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback

file_dir = 'C:/Users/Izzy/Question_Detection_And_Testing/IsraelFiles/1526311378.0438612'
new_csv = 'new_transcripts.csv'
speech_to_text = SpeechToTextV1(
	iam_api_key='LpSYesWzDWW3CLtik3wHiHzfRujpyHPKCEKrisqRcSia',
    url='https://gateway-wdc.watsonplatform.net/speech-to-text/api')

all_transcripts = {}

'''
class MyRecognizeCallback(RecognizeCallback):
    def __init__(self, filename):
    	RecognizeCallback.__init__(self)
    	self.filename = filename
    	all_transcripts[self.filename] = list()

    def on_transcription(self, transcript):
        speech_result = transcript[0]
        transcript = speech_result['transcript']
        #print(transcript)

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_transcription_complete(self):
        print('Transcription completed')

    def on_hypothesis(self, hypothesis):
        #print(hypothesis)
        pass

    def on_data(self, data):
        speech_result = data['results'][0]
        transcript = speech_result['alternatives'][0]['transcript']
        print(data)
        all_transcripts[self.filename].append(transcript)
'''

def file_maker():
	for name in os.listdir(file_dir):
		if ('%' not in name and '.wav' in name) and ((len(name.split('_')) == 1) and 'cleanup' not in name):
			ffmpeg_call = 'ffmpeg -i ' + name + ' -ac 1 ' + name.split('.')[0] + '_mono.wav'
			subprocess.call(ffmpeg_call)

def google_transcribe(speech_file):
    client = speech.SpeechClient()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.types.RecognitionAudio(content=content)
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True)

    response = client.recognize(config, audio)
    returned_item = list()
    time_stamp_collect = list()
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        returned_item.append(alternative.transcript)
        total_time = 0.0
        counter = -1
        if len(alternative.words) > 1:
        	endstamp = alternative.words[-1].end_time
        	endstamp_time = endstamp.seconds + endstamp.nanos * 1e-9
        	while(total_time <= 0.5):
        		word_start = alternative.words[counter].start_time
        		start_time = word_start.seconds + word_start.nanos * 1e-9
        		total_time = endstamp_time - start_time
        		if alternative.words[counter] != alternative.words[0]:
        			counter -=1
        		else:
        			break
        	time_stamps = (start_time, endstamp_time)
        	time_stamp_collect.append(time_stamps)
        elif len(alternative.words) == 1:
        	endstamp = alternative.words[-1].end_time
        	endstamp_time = endstamp.seconds + endstamp.nanos * 1e-9
        	word_start = alternative.words[counter].start_time
        	start_time = word_start.seconds + word_start.nanos * 1e-9
        	total_time = endstamp_time - start_time
        	time_stamps = (start_time, endstamp_time)
        	time_stamp_collect.append(time_stamps)
        else:
        	time_stamp_collect.append(0,0)
    return returned_item, time_stamp_collect

def google_repeater():
	#file_maker()
	with open('new_transcripts.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter='\t', quotechar='|', lineterminator = '\n')
		dir_list = list()
		master_list = list()
		for name in os.listdir(file_dir):
			if 'mono.wav' in name:
				dir_list.append(name)
		sorted_dir_list = sorted(dir_list, key=lambda x: int(x.split('_')[0]))
		for piece in sorted_dir_list:
			iterable = google_transcribe(piece)
			if len(iterable[0]) > 1:
				for num in range(len(iterable)):
					tran = iterable[0]
					stamp = iterable[1]
					label = int('?' in tran[num])
					master_list.append([piece, tran[num], stamp[num], label])
			else:
				tran = iterable[0]
				stamp = iterable[1]
				if tran != []:
					label = int('?' in tran[0])
					master_list.append([piece, tran[0], stamp[0], label])
				else: 
					label = 0
					master_list.append([piece, '', (0,0), label])
		writer.writerows(master_list)
		print(master_list)

'''
def ibm_repeater():
	#file_maker()
	ibm_list = list()
	dir_list = list()
	with open('ibm_transcripts.csv', 'w') as csvfile:
		master_list =list()
		writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n')
		for name in os.listdir(file_dir):
			if 'mono.wav' in name:
				dir_list.append(name)
		sorted_dir_list = sorted(dir_list, key=lambda x: int(x.split('_')[0]))
		for bit in sorted_dir_list:
			with open(bit, 'rb') as audio_file:
				speech_recognition_results = speech_to_text.recognize(
					audio=audio_file,
					content_type='audio/wav',
					interim_results=False,
					)
			full_transcript = ''
			for alt in speech_recognition_results["results"]:
				for dictionary in alt["alternatives"]:
					master_list.append([bit,dictionary['transcript']])
		writer.writerows(master_list)

def csv_combine():
	ibm = open('ibm_transcripts.csv', 'r')
	google = open('new_transcripts.csv','r')
	i_reader = csv.reader(ibm,delimiter='\t')
	g_reader = csv.reader(google,delimiter='\t')
	master_i = list()
	for row in i_reader:
		temp_i = list()
		for item in row:
			temp_i.append(item)
		master_i.append(temp_i)
	master_g = list()
	for row in g_reader:
		temp_g = list()
		for item in row:
			temp_g.append(item)
		master_g.append(temp_g)
	with open('combined_transcripts.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n')
		for num in range(len(master_g) - 1):
			temp_n = list()
			addition = master_i[num] + master_g[num][1:3]
			writer.writerow(addition)
	ibm.close()
	google.close()
'''
file_maker()