import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.models import Sequential, load_model
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
from sklearn.metrics import confusion_matrix

from funkcije import prikazi, prilagodi, sorensen_dice_coefficient, plotsaving

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
			sol.append(prilagodi(backup[:, left:right]))
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
	prikazi("Histogram", arrimg)
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
	"""
	#Segmentacija na linije
	prikazi("Segmentacija na linije", original)
	"""
	
	chars = []
	frames = []
	for i in range(0, len(lines)): 
		if lines[i-1] >= lines[i]:
			#print("dalje") 
			continue
		charseg_output = charseg(original1[lines[i-1]:lines[i], :], lines[i-1], lines[i]) #nakon segmentacije po histogramu
		#prikazi("Slika", original1[lines[i-1]:lines[i], :])
		chars += charseg_output[0]
		frames += charseg_output[1]
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
		letter = line[-5]
		if letter in ['(i)je', 'ja, (i)je', 'ja,(i)je']:
			letter = 'ja, (i)je'
		gclass.append(letter)
		line = [int(float(x)) for x in line[-4:]]
		cv.rectangle(img,(line[0],line[1]),(line[2], line[3]),(0,0,255),1)
		seg.append(prilagodi(img[line[1]:line[3], line[0]:line[2]]))
		frames.append((line[1], line[3], line[0], line[2]))
	
	return (seg, frames, gclass)

def test_segmentation(img, truth_frames, guess_frames, model_name, file):
	frame_mapping = []
	for gue in guess_frames:
		found = 0
		possible_frames = []
		sdc_list = []
		for tr in truth_frames:
			sdc = sorensen_dice_coefficient(gue, tr)
			if (sdc >= 0.70):
				possible_frames.append(tr)
				sdc_list.append(sdc)
				#print(sdc, gue, tr)
				found += 1
				break
		if found == 0:
			frame_mapping.append(None)
		else:
			maksi = max(sdc_list)
			maksind = sdc_list.index(maksi)
			frame_mapping.append(possible_frames[maksind])
	file.write(model_name + '\n')
	file.write(img + " Stvaran broj okvira: " + str(len(truth_frames)) + '\n')
	file.write(str(len(frame_mapping) - frame_mapping.count(None)) + ' od ' + str(len(frame_mapping)) + ' izrezanih su istiniti.\n')
	return frame_mapping

def show_correctly_mapped_frames(page_img_loc, frame_mapping, true_mapping):
	kopija = cv.imread(page_img_loc, 0)
	for tm in true_mapping:
		cv.line(kopija, (tm[2], tm[0]), (tm[2], tm[1]), 255, 1)
		cv.line(kopija, (tm[3], tm[0]), (tm[3], tm[1]), 255, 1)
		cv.line(kopija, (tm[2], tm[0]), (tm[3], tm[0]), 255, 1)
		cv.line(kopija, (tm[2], tm[1]), (tm[3], tm[1]), 255, 1)
	for fm in frame_mapping:
		if fm == None:
			continue
		cv.line(kopija, (fm[2], fm[0]), (fm[2], fm[1]), 1, 1)
		cv.line(kopija, (fm[3], fm[0]), (fm[3], fm[1]), 1, 1)
		cv.line(kopija, (fm[2], fm[0]), (fm[3], fm[0]), 1, 1)
		cv.line(kopija, (fm[2], fm[1]), (fm[3], fm[1]), 1, 1)
	prikazi("", kopija)	
	
def get_conf_matrix(loaded_model, truth_class, segments, azbuka):
	ConfMatrix = [[0 for x in range(num_classes)] for y in range(num_classes)]

	test_labels = []
	rounded_predictions = []

	X = [np.array(im).flatten() for im in segments]

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
	
	num = 0 #tocni
	total = 0 #svi
	#print(len(output), len(truth_class), len(segments))
	if len(output) > 0:
		for i in range(len(output)):
			#slovo koje je ispalo
			indeks = output[i].argmax()
			#slovo koje bi trebalo ispasti
			slovo = truth_class[i]
			#if slovo == '(i)je':
			#	continue
				#slovo = 'ja, (i)je'
			test_labels.append(slovo)
			#print(azbuka[indeks], slovo)
			ConfMatrix[azbuka.index(slovo)][indeks] += 1 
			rounded_predictions.append(azbuka[indeks])
			total += 1
			if azbuka[indeks] == slovo:
				num += 1
	#print(num, total)
	#print(' '.join(gclass)) #tocan tekst
	cm = confusion_matrix(test_labels, rounded_predictions, azbuka)
	return (cm, test_labels, rounded_predictions)

def statistic_saving(cm, stat_loc, azbuka, test_labels, rounded_predictions):
	file = open(stat_loc,'w') 
	acc_by_class = []
	acc = []
	file.write('Statistics from the confusion matrix\n')
	for i in range(len(cm)):
		correct = cm[i][i]
		all = sum(cm[i])
		if all == 0: #izbjegavanje nan vrijednosti
			continue
		acc_by_class.append((correct / all,  i))
		acc.append((correct / all,  i))
		file.write("Slovo " + azbuka[i] + " TP " + str (cm[i][i]) + " Total " +str(sum(cm[i])) + '\n')

	acc_by_class.sort()
	acc_by_class.reverse()
	file.write('---------------\n')
	file.write('Five most accurate classes: \n')
	for a in acc_by_class[:5]:
		file.write(azbuka[a[1]] + ' ' + str(a[0]) + ' ' + str(cm[a[1]]) + '\n')
	file.write('---------------\n')
	file.write('Total accuracy: ' + str(accuracy_score(test_labels, rounded_predictions, normalize=True)) + '\n')
	#file.write('Total accuracy2 ' + str(tocno / (tocno + krivo)) + '\n')
	file.write('---------------\n')
	file.write('Accuracy by class: \n')
	for a in acc:
		file.write(azbuka[a[1]] + ' ' + str(a[0]) + '\n')
	file.close()
	
def testiraj(model_loc,  truth_frames, truth_class, truth_seg, guess_seg, guess_frames, frame_mapping, plot_loc, stat_loc, azbuka, f, page_img_loc):
	'''
	#ucitaj json i stvori model
	json_file = open(model_json_loc, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#ucitaj tezine za model
	loaded_model.load_weights(model_weights_loc)
	'''
	loaded_model = load_model(model_loc)
	
	(cm, test_labels, rounded_predictions) = get_conf_matrix(loaded_model, truth_class, truth_seg, azbuka)
	plotsaving(cm, azbuka, plot_loc, normalize=True, title='Matrica zabune')
	statistic_saving(cm, stat_loc, azbuka, test_labels, rounded_predictions)
	
	guess_seg = np.reshape(guess_seg, (-1, 50, 50, 1))
	all_predictions = loaded_model.predict(guess_seg)
	s = []
	for l, img in zip(all_predictions, guess_seg):
		s.append(azbuka[l.argmax()])
		#prikazi(str(azbuka[l.argmax()]), img)
	f.write(page_img_loc + '\n')
	f.write(model_loc + '\n')
	f.write(' '.join(s) + '\n')
	f.write('------------------------------------------------------------------------------------------\n')
	"""
	reduced_truth_class = []
	reduced_guess_seg = []
	for i in range(len(frame_mapping)):
		if frame_mapping[i] == None:
			continue
		else:
			j = truth_frames.index(frame_mapping[i])
			reduced_truth_class.append(truth_class[j])
			reduced_guess_seg.append(guess_seg[i])
		print(reduced_truth_class, reduced_guess_seg)

	cm = get_conf_matrix(loaded_model, reduced_truth_class, reduced_guess_seg, azbuka)
	"""
	#plotsaving(cm, azbuka, plot_loc, normalize='False', title='Confusion matrix')

azbuka = ['a', 'b', 'v', 'g', 'd', 'e', 'zj', 'dz', 'z', '(i)', 'i', 'dj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'h', '(o)', "(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je', 'ju' ,'j', 'poluglas']
azbuka.sort()
img_rows, img_cols = 50, 50
num_classes = 33

all_models = ["modelLeNet", "modelarapski", "modelkineski1", "modelkineski2"]
#all_models = ["modelarapski"]

test_seg_file = open('Rezultati/Test segmetation.txt','w') 
content_file = open('Rezultati/Content_file.txt','w') 
for mod in all_models:
	for dataset in range(1, 4):
		model_loc = 'Modeli/' + mod + '/' + mod + '-data' + str(dataset) + '.h5'
		tests = os.listdir('Testovi')
		for title in tests:
			folders = os.listdir('Testovi/' + title)
			
			truth_seg_for_folder = []
			truth_frames_for_folder = []
			truth_class_for_folder = []
			guess_seg_for_folder = []
			guess_frames_for_folder = []
			frame_mapping_for_folder = []
			
			for folder in folders:
				page_img_loc = 'Testovi/' + title + '/' + folder + '/' + str.lower(title) + folder + '.png'
				page_txt_loc = 'Testovi/' + title + '/' + folder + '/' + str.lower(title + folder) + '.txt'

				(truth_seg, truth_frames, truth_class) = get_truth(page_img_loc, page_txt_loc)
				(guess_seg, guess_frames) = split_to_lines(page_img_loc)
				
				frame_mapping = test_segmentation(page_img_loc, truth_frames, guess_frames, mod + '-data' + str(dataset), test_seg_file)
				#show_correctly_mapped_frames(page_img_loc, frame_mapping, truth_frames)
				#show_correctly_mapped_frames(page_img_loc, guess_frames, truth_frames)
			
				truth_seg_for_folder += truth_seg
				truth_frames_for_folder += truth_frames
				truth_class_for_folder += truth_class
				guess_seg_for_folder += guess_seg
				guess_frames_for_folder += guess_frames
				frame_mapping_for_folder += frame_mapping
			
				#Testiranje tocnosti
			plot_loc = 'Rezultati/' + mod + '/' + mod + str.lower(title) + '-data' + str(dataset) + '.png'
			stat_loc = 'Rezultati/' + mod + '/' + mod + str.lower(title) + '-data' + str(dataset) + '.txt'
			testiraj(model_loc, truth_frames_for_folder, truth_class_for_folder, truth_seg_for_folder, guess_seg_for_folder, guess_frames_for_folder, frame_mapping_for_folder, plot_loc, stat_loc, azbuka, content_file, page_img_loc)

test_seg_file.close()
content_file.close()