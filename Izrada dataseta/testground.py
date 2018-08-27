import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
import os
from sklearn.utils import shuffle
import shutil
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import accuracy_score

from funkcije import prikazi, prilagodi

from math import floor

def testiraj(model_json_loc, model_weights_loc, page_img_loc, page_txt_loc):
	#ucitaj json i stvori model
	json_file = open(model_json_loc, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#ucitaj tezine za model
	loaded_model.load_weights(model_weights_loc)

	#ucitavanje slike koja se analizira
	img = cv.imread(page_img_loc, 0)
	ret, img = cv.threshold(img,127,255,cv.THRESH_BINARY)
	file = open(page_txt_loc, "r")

	#segmenti izvuceni iz slike
	seg = []

	#temejna istina za slike, koja slova stvarno prikazuju
	gclass = []

	for line in file: 
		line = line.split()
		gclass.append(line[-5])
		line = [int(float(x)) for x in line[-4:]]
		cv.rectangle(img,(line[0],line[1]),(line[2], line[3]),(0,0,255),1)
		seg.append(prilagodi(img[line[1]:line[3], line[0]:line[2]]))

	ConfMatrix = [[0 for x in range(num_classes)] for y in range(num_classes)]

	rounded_predictions = []

	X = [np.array(im).flatten() for im in seg]

	X = np.array(X)

	samples = len(X)

	if K.image_data_format() == 'channels_first':
		X = X.reshape(samples, 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X = X.reshape(samples, img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	X = X.astype('float32')

	X /= 255

	output = loaded_model.predict(X, verbose=1)

	num = 0
	total = 0
	if len(output) > 0:
		for i in range(len(output)):
			#slovo koje je ispalo
			indeks = output[i].argmax()
			#slovo koje bi trebalo ispasti
			slovo = gclass[i]
			if slovo == '(i)je':
				continue
			print(azbuka[indeks], slovo)
			ConfMatrix[azbuka.index(slovo)][indeks] += 1 
			rounded_predictions.append(azbuka[indeks])
			total += 1
			if azbuka[indeks] == slovo:
				num += 1
	print(num, total)
	#print(' '.join(gclass))

azbuka = ['a', 'b', 'v', 'g', 'd', 'e', 'zj', 'dz', 'z', '(i)', 'i', 'dj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'h', '(o)', "(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je', 'ju' ,'j', 'poluglas']
azbuka.sort()
img_rows, img_cols = 50, 50
num_classes = 33

all_models = ["modelLeNet", "modelarapski", "modelkineski1", "modelkineski2"]
for mod in all_models:
	print("Model " + mod + " je ucitan")
	model_json_loc = 'Modeli/' + mod + '/' + mod + '-data1.json'
	model_weights_loc = 'Modeli/' + mod + '/' + mod + '-data1.h5'
	tests = os.listdir('Testovi')
	for title in tests:
		folders = os.listdir('Testovi/' + title)
		for folder in folders:
			page_img_loc = 'Testovi/' + title + '/' + folder + '/' + str.lower(title) + folder + '.png'
			page_txt_loc = 'Testovi/' + title + '/' + folder + '/' + str.lower(title + folder) + '.txt'

			testiraj(model_json_loc, model_weights_loc, page_img_loc, page_txt_loc)