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
from copy import deepcopy
import matplotlib.pyplot as plt

def flood(c, y, x):
	h, w = c.shape
	if y < 0 or x < 0 or y >= h or x >= w:
		return 
	if (c[y, x] > 0):
		c[y, x] = 0
	else:
		return 
	flood(c, y - 1, x)
	flood(c, y + 1, x)
	flood(c, y, x - 1)
	flood(c, y , x + 1)

def slices(c):
	#c = cv.adaptiveThreshold(c,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
	c = cv.adaptiveThreshold(c,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
	#c = cv.GaussianBlur(c,(5,5),0)
	#ret, c = cv.threshold(c,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
	c = cv.bitwise_not(c)
	backup = deepcopy(c)
	"""
	prikazi("Slika", c)
	"""
	h, w = c.shape
	add_slice = []
	last = 0
	for i in range(w):
		if c[h//2, i] > 0:
			flood(c, h//2, i)
			add_slice.append(backup[:, last:i])
			last = i
			"""
			prikazi("Slika", c)
			"""
	add_slice.append(backup[:, last:])
	return add_slice

def charseg(row_img, top, bottom):
	sol = []
	frames = []
	h, w = row_img.shape
	backup = deepcopy(row_img)
	row_img = cv.GaussianBlur(row_img,(5,5),0)
	ret, row_img = cv.threshold(row_img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
	row_img = cv.bitwise_not(row_img)

	kernel = np.ones((2, 2),np.uint8)
	row_img = cv.erode(row_img,kernel,iterations = 2)
	arrimg = np.array(row_img)
	
	hist_col = arrimg.sum(axis=0)
	"""
	plt.hist(hist_col, bins=30)
	plt.ylabel('Pikseli s tintom');
	plt.show()
	"""
	bef = 0
	left = 0
	right = 0
	for i in range(1, len(hist_col)): # od 1 da left ne ode izvan polja
		if hist_col[i] == 0 and bef > 0:
			right = i
			sol.append(backup[:, left:right])
			frames.append((top, bottom, left, right))
			#cv.line(backup, (right, 0), (right , h), 255, 1)
			#cv.line(backup, (left, 0), (left , h), 1, 1)
		if hist_col[i] > 0 and bef == 0:
			left = i-1
		bef = hist_col[i]
	"""
	prikazi("Slika", row_img)
	
	prikazi("Slika", backup)
	print(hist_col)
	"""
	return (sol, frames)

def place(li, lines, hist, tol, hi, lo):
	if li < lo or li > hi:
		return -1
	li = np.argmin(hist[max(li-tol, lo):min(li + tol, hi)]) + li-tol
	for l in lines:
		if abs(l - li) < tol:
			return -1
		if li == l:
			return -1
	return li
	
def split_to_lines(img_loc):
	original = cv.imread(img_loc,0)
	img = cv.GaussianBlur(original,(5,5),0)
	ret, img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

	img = cv.bitwise_not(img)

	kernel = np.ones((3, 3),np.uint8)
	img = cv.erode(img,kernel,iterations = 2)
	arrimg = np.array(img)
	hist_row = arrimg.sum(axis=1)

	"""
	plt.hist(hist_row, bins=30)
	plt.ylabel('Pikseli s tintom');
	plt.show()

	print(hist_row)
	prikazi("Slika", arrimg)
	"""

	h, w = original.shape
	original1 = deepcopy(original)
	lines = []
	bef = 0
	for i in range(len(hist_row)):
		if hist_row[i]/255/w < 0.1 and bef > 0.1:
			lines.append(i)
		bef = hist_row[i]/255/w
	#print(lines)
		
	avg = h // (len(lines))
	#print("Avg:", avg)
	lines = [0]
	bef = 0
	for i in range(len(hist_row)):
		if hist_row[i]/255/w == 0 and bef > 0.1:
			if lines[-1] + avg//3 <= i:
				lines.append(i)
				cv.line(original, (0, i), (w, i), 1, 1)
		bef = hist_row[i]/255/w

	lineThickness = 1
	prom = True
	while (prom):
		prom = False
		for i in range(len(lines)):
			#prava linija
			li = lines[i]
			#print("Linija", i + 1, " -> ", li)
			cv.line(original, (0, li), (w, li), 1, lineThickness)
			
			#linija iznad sadasnje		
			ind = place(li - avg, lines, hist_row, avg//3, h, 0)
			if ind != -1:
				lines.append(ind)
				lines.sort()
				cv.line(original, (0, ind), (w, ind), 255, lineThickness)
				prom = True
				break
			#linija ispod sadasnje
			ind = place(li + avg, lines, hist_row, avg//3, h, 0)
			if ind != -1:
				lines.append(ind)
				lines.sort()
				cv.line(original, (0, ind), (w, ind), 255, lineThickness)
				prom = True
				break

	prikazi("Slika", original)

	chars = []
	frames = []
	for i in range(1, 2): #TODO stavi sve linije a ne 1
		if lines[i-1] >= lines[i]:
			print("dalje") 
			continue
		charseg_output = charseg(original1[lines[i-1]:lines[i], :], lines[i-1], lines[i]) #nakon segmentacije po histogramu
		chars.append(charseg_output[0])
		frames.append(charseg_output[1])
		"""
		seg = charseg_output[0]
		for c in seg:
			sl = slices(c) #dodatno sjeckanje slova
			print("Slices: ", len(sl))
			for s in sl:
				hh, ww = s.shape
				if hh == 0 or ww == 0:
					continue
				
				prikazi("Slika", s)
				
				chars.append(cv.bitwise_not(s)) #!!!!!
		"""
	return (chars, frames)

def get_truth(page_img_loc, page_txt_loc):
	#ucitavanje slike koja se analizira
	img = cv.imread(page_img_loc, 0)
	ret, img = cv.threshold(img,127,255,cv.THRESH_BINARY)
	file = open(page_txt_loc, "r")

	#segmenti izvuceni iz slike
	seg = []

	#temejna istina za slike, koja slova stvarno prikazuju
	gclass = []

	#lista okvira svih slika
	frames = []
	
	for line in file: 
		line = line.split()
		gclass.append(line[-5])
		line = [int(float(x)) for x in line[-4:]]
		cv.rectangle(img,(line[0],line[1]),(line[2], line[3]),(0,0,255),1)
		seg.append(prilagodi(img[line[1]:line[3], line[0]:line[2]]))
		frames.append((line[1], line[3], line[0], line[2]))
	
	return (seg, frames, gclass)
	
def testiraj(model_json_loc, model_weights_loc, truth_frames, truth_class, guess_seg, guess_frames):
	#ucitaj json i stvori model
	json_file = open(model_json_loc, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#ucitaj tezine za model
	loaded_model.load_weights(model_weights_loc)

	ConfMatrix = [[0 for x in range(num_classes)] for y in range(num_classes)]

	rounded_predictions = []

	X = [np.array(im).flatten() for im in guess_seg]

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
			slovo = truth_class[i]
			if slovo == '(i)je':
				continue
			print(azbuka[indeks], slovo)
			ConfMatrix[azbuka.index(slovo)][indeks] += 1 
			rounded_predictions.append(azbuka[indeks])
			total += 1
			if azbuka[indeks] == slovo:
				num += 1
	#print(num, total)
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

			(truth_seg, truth_frames, truth_class) = get_truth(page_img_loc, page_txt_loc)
			(guess_seg, guess_frames) = split_to_lines(page_img_loc)
			test_segmentation(truth_frames, guess_frames)
			#testiraj(model_json_loc, model_weights_loc, truth_frames, truth_class, guess_seg, guess_frames)