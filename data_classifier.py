import os
import csv
import math
import sklearn
import wave
import subprocess
import parselmouth
import numpy as np
from respeaker_questions import extract_prosodic_features, check_question_words, extract_general_stats, get_contour_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import svm

root = 'C:/Users/Izzy/Question_Detection_And_Testing/'
sound_dir_1 = 'C:/Users/Izzy/Question_Detection_And_Testing/IsraelFiles/1527174736.6341195/'
csv_q = 'C:/Users/Izzy/Question_Detection_And_Testing/IsraelFiles/1527174736.6341195/new_transcripts.csv'
sound_dir_2 = 'C:/Users/Izzy/Question_Detection_And_Testing/IsraelFiles/1526311378.0438612/'
csv_a = 'C:/Users/Izzy/Question_Detection_And_Testing/IsraelFiles/1526311378.0438612/new_transcripts.csv'

def file_maker(sound_dir):
	os.chdir(sound_dir)
	for name in os.listdir(sound_dir):
		if ('%' not in name and '.wav' in name) and ((len(name.split('_')) == 1) and 'cleanup' not in name):
			ffmpeg_call = 'ffmpeg -i ' + name + ' -ac 1 ' + name.split('.')[0] + '_mono.wav'
			subprocess.call(ffmpeg_call)
	os.chdir(root)


def wave_snip(csv_file, sound):
	file_maker(sound)
	with open(csv_file,'r') as csvfile:
		reader = csv.reader(csvfile, delimiter='\t', lineterminator='\n')
		stamp_list = [(row[0],row[2]) for row in reader if row[1] != '']
	file_list = list()
	for count, value in enumerate(stamp_list):
		new_name = '{}utterance.wav'.format(count+1)
		loud_name = '{}utteranceloud.wav'.format(count+1)
		new_dir = sound + loud_name
		pair = stamp_list[count]
		file_list.append(new_dir)
		file = pair[0]
		stamp = eval(pair[1])
		start_time = stamp[0]
		end_stamp = stamp[1]
		duration = end_stamp - start_time
		ffmpeg_call_1 = 'ffmpeg -ss {} -i'.format(start_time) + ' {} -to '.format(file) + '{} '.format(end_stamp) + '{}'.format(new_name)
		ffmpeg_call_2 = 'ffmpeg -i {} -filter:a "volume=1.5" {}'.format(new_name, loud_name)
		os.chdir(sound)
		subprocess.call(ffmpeg_call_1)
		subprocess.call(ffmpeg_call_2)
		os.chdir(root)
	return file_list

def import_data(csv_file, sound):
	sound_list = wave_snip(csv_file, sound)
	with open(csv_file, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter='\t', lineterminator='\n')
		utterance_list = list()
		label_list = list()
		for row in reader:
			if row[1] != '':
				utterance_list.append(row[1])
				label_list.append(row[3])
	lexical_features = list()
	for string in utterance_list:
		quest_indices, inquiry_indices, auxverb_indices = check_question_words(string)
		c_data = [int(0 in quest_indices)]
		if quest_indices:
			c_data.append(quest_indices[0])
		else:
			c_data.append(-1)
		if inquiry_indices:
			c_data.append(inquiry_indices[0])
		else:
			c_data.append(-1)
		lexical_features.append(c_data)
	master_feature_list = list()
	counter = 0
	for file in sound_list:
		feature_list = list()
		p, fo_slope_vals, fo_slope, fo_end_vals, o_fos = extract_prosodic_features(file)
		if fo_end_vals != None:
			#feature_list += extract_general_stats(fo_end_vals)
			feature_list += get_contour_features(fo_end_vals)
			#feature_list.append(fo_slope)
			feature_list += lexical_features[counter]
			feature_list.append(int(label_list[counter]))
			master_feature_list.append(feature_list)
		counter += 1
	cleanup(sound)
	return master_feature_list

def entrainment(csv_file, sound):
	file_list = wave_snip(csv_file, sound)
	master_list = list()
	for file in file_list:
		snd = parselmouth.Sound(file)
		intensity = snd.to_intensity()
		pitch = snd.to_pitch()
		pitch_list = [pitch.get_value_in_frame(frame) for frame in range(pitch.get_number_of_frames())]
		intensity_list = intensity.as_array()[0]
		master_list.append([file.split('/')[-1], pitch_list, intensity_list])
	cleanup(sound)
	return master_list

def cleanup(direc):
	print(direc)
	os.chdir(direc)
	for filename in os.listdir(direc):
		if 'mono' in filename and '.py' not in filename:
			os.remove(filename)
		if 'utterance.wav' in filename:
			os.remove(filename)
		if 'loud.wav' in filename:
			os.remove(filename)
	os.chdir(root)

#print(entrainment(csv_q, sound_dir_1))


X = import_data(csv_q, sound_dir_1) + import_data(csv_a, sound_dir_2)

train, test = train_test_split(X, test_size = 0.2, shuffle=True)

label_col = len(train[0]) - 1 

train = np.array(train)
test = np.array(test)

clf = svm.SVC(kernel='linear', C=0.8)
clf = clf.fit(train[:,0:label_col], train[:,label_col])
res = clf.predict(test[:,0:label_col])
NB_for_p = metrics.precision_score(test[:,label_col], res)
NB_for_r = metrics.recall_score(test[:,label_col], res)
print("{} Precision: ".format(''), NB_for_p)
print("{} Recall: ".format(''), NB_for_r)
