import csv
import wave
import os
import numpy as np
import sklearn
from respeaker_questions import extract_prosodic_features, check_question_words, extract_general_stats, get_contour_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import tree


question_transcripts = 'question_transcripts.csv'
question_directory = 'C:/Users/Izzy/Question_Detection_And_Testing/IsraelFiles/1527174736.6341195'

answer_transcripts = 'answer_transcripts.csv'
answer_directory = 'C:/Users/Izzy/Question_Detection_And_Testing/IsraelFiles/1526311378.0438612'

# Build list of full utterances
def create_utterance_list(our_file):
	with open(our_file, newline='') as csvfile:
		question_reader = csv.reader(csvfile, delimiter= '\t')
		word_collect = list()
		for row in question_reader:
			word_collect.append(row[0])
		return word_collect

# Build sorted list of filenames
def build_file_list(directory):
	file_list = list()
	for filename in os.listdir(directory):
		if (filename[0].isalpha() or '%' in filename) and '.wav' in filename:
			file_list.append(filename)
	file_list_sorted = sorted(file_list, key=lambda x: os.path.getmtime(directory + '/' + x))
	#Use file list as a stack
	return file_list_sorted[::-1]

#Group wav files according to utterances
def create_wave_list(wave_directory, csv_select):
	utterances = create_utterance_list(csv_select)
	files = build_file_list(wave_directory)
	wave_list = list()
	lexical_features = list()
	for string in utterances:
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
		temp_list = list()
		temp_list_sorted = list()
		for word in string.split():
			if len(files) != 0:
				if '%' not in files[-1]:
					temp_list.append(files.pop())
				else:
					files.pop()
		temp_list_sorted = sorted(temp_list, key=lambda x: int(x.split('_')[1][:-4]))
		if temp_list_sorted != []:
			wave_list.append(temp_list_sorted)
	new_dir_list = wave_stitch(wave_list, wave_directory)
	return new_dir_list, lexical_features

def wave_stitch(name_list, directory):
	new_dir = directory + '/fusion sound'
	new_dir_list = list()
	counter = 1
	for group in name_list:
		with wave.open(new_dir + '/utterance' + str(counter) + '.wav', mode='w') as new_file:
			if directory == answer_directory:
				with wave.open(directory + '/all_1215.wav') as pf:
					new_file.setparams((pf.getnchannels(),pf.getsampwidth(),pf.getframerate(),pf.getnframes(),pf.getcomptype(),pf.getcompname()))
			else:
				with wave.open(directory + '/all_1738400.wav') as pf:
					new_file.setparams((pf.getnchannels(),pf.getsampwidth(),pf.getframerate(),pf.getnframes(),pf.getcomptype(),pf.getcompname()))
			for name in group:
				with wave.open(directory + '/' + name) as old_file:
					frames = old_file.readframes(old_file.getnframes())
					new_file.writeframesraw(frames)
			new_dir_list.append(new_dir + '/utterance' + str(counter) + '.wav')
		counter += 1
	return new_dir_list

print(create_wave_list(question_directory,question_transcripts))

'''
def compile_features(wave_dir, lexical_features):
	master_feature_list = list()
	counter = 0
	for utterance in wave_dir:
		if utterance != []:
			feature_list = list()
			p, fo_slope_vals, fo_slope, fo_end_vals, o_fos = extract_prosodic_features(utterance)
			if fo_end_vals != None:
				feature_list += extract_general_stats(fo_end_vals)
				feature_list += get_contour_features(fo_end_vals)
				feature_list.append(fo_slope)
				feature_list += lexical_features[counter]
				if '1526311378.0438612' in utterance:
					feature_list.append(0)
				else:
					feature_list.append(1)
				master_feature_list.append(feature_list)
		os.remove(utterance)
		counter += 1
	return master_feature_list

q_wave_list, q_lexical_features = create_wave_list(question_directory, question_transcripts)
a_wave_list, a_lexical_features = create_wave_list(answer_directory, answer_transcripts)

X = compile_features(q_wave_list, q_lexical_features) + compile_features(a_wave_list, a_lexical_features)

train, test = train_test_split(X, test_size = 0.2, shuffle=True)

label_col = len(train[0]) - 1 

train = np.array(train)
test = np.array(test)

#print(train[:,14])	

clf = AdaBoostClassifier()
clf = clf.fit(train[:,0:label_col], train[:,label_col])
correct = 0
res = clf.predict(test[:,0:label_col])
NB_for_p = metrics.precision_score(test[:,label_col], res)
NB_for_r = metrics.recall_score(test[:,label_col], res)
print("Precision: ", NB_for_p)
print("Recall: ", NB_for_r)

'''