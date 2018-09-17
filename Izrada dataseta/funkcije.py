#cesto koristene funkcije

import os
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

alphabet_for_display = {'a': 'a', 'b': 'b', 'v': 'v', 'g': 'g', 'd': 'd', 'e': 'e', 'zj': u' ž', 'dz': 'dž', 'z': 'z', '(i)': '(i)', 'i': 'i', 'dj': 'đ', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'r': 'r', 's': 's', 't': 't', 'u': 'u', 'f': 'f', 'h': 'h', '(o)': '(o)', "(sj)c'": '(š)ć', 'c': 'c', 'cj': 'č', 'sj': 'š', 'ja, (i)je': 'ja, (i)je', 'ju': 'ju','j': 'j', 'poluglas': 'poluglas'}

def prilagodi(img):
    color = [255,0,0]
    imgsize = img.shape
    w = imgsize[0]
    h = imgsize[1]
    gore = 0
    dolje = 0
    lijevo = 0
    desno = 0
    if h > w:
        diff = h - w
        lijevo = int(diff / 2) + (diff % 2)
        desno = int(diff / 2)
    else:
        diff = w - h
        gore = int(diff / 2) + (diff % 2)
        dolje = int(diff / 2)
    img = cv.copyMakeBorder(img, lijevo, desno, gore, dolje, cv.BORDER_CONSTANT, value=color)
    img = cv.resize(img, (50, 50), interpolation=cv.INTER_AREA)
    return img
	
def prikazi(title, image):
	cv.imshow(title, image)
	cv.waitKey(0)
	cv.destroyAllWindows()
			
def noisy(noise_list,image):
	row,col= image.shape
	ch = 1
	for noise_type in noise_list:
		if noise_type == "gauss":
			mean = 0
			var = 0.1
			sigma = var**0.5
			gauss = np.random.normal(mean,sigma,(row,col,ch))
			gauss = gauss.reshape(row,col,ch)
			cv.randn(gauss,(0),(99))
			noisy = image
			for r in range(row):
				for c in range(col):
					noisy[r, c] = max(gauss[r, c] + image[r, c], noisy[r, c])
			image = noisy
		elif noise_type == "s&p":
			s_vs_p = 0.5
			amount = 0.04
			out = np.copy(image)
			# Salt mode
			num_salt = np.ceil(amount * image.size * s_vs_p)
			coords = [np.random.randint(0, i - 1, int(num_salt))
				  for i in image.shape]
			out[coords] = 255

			# Pepper mode
			num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
			coords = [np.random.randint(0, i - 1, int(num_pepper))
				  for i in image.shape]
			out[coords] = random.randint(0, 150)
			image =  out
		elif noise_type == "poisson":
			vals = len(np.unique(image))
			vals = 2 ** np.ceil(np.log2(vals))
			noisy = np.random.poisson(image * vals) / float(vals)
			for r in range(row):
				for c in range(col):
					image[r, c] = noisy[r, c]
		elif noise_type =="speckle":
			gauss = np.random.rand(row,col)
			gauss = gauss.reshape(row,col)        
			noisy = image + image * gauss
			for r in range(row):
				for c in range(col):
					image[r, c] = noisy[r, c]
	return image
	
def plotsaving(cm, classes, dst, normalize='False', title='Matrica zabune'):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig = plt.figure(figsize=(10, 10), dpi=300)
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(cm), cmap=plt.cm.jet, 
					interpolation='nearest')
	plt.title(title)

	width = len(classes)
	height = len(classes)

	cb = fig.colorbar(res)
	
	alphabet = []
	for i in range(len(classes)):
		alphabet.append(' ' + alphabet_for_display[classes[i]])
	
	
	plt.xticks(range(width), alphabet[:width], rotation=90)
	plt.yticks(range(height), alphabet[:height])
	plt.tight_layout(pad=1.5)
	plt.ylabel("Stvarna oznaka")
	plt.xlabel("Predviđena oznaka")
	plt.savefig(dst, format='png')

#TODO: Remove this function
def plot_confusion_matrix(cm, classes, normalize='False', title='Matrica zabune', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalizirana matrica zabune")
	else:
		print("Matrica zabune bez normalizacije")
	print(cm)
	tresh = cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j,i, '', 
		horizontalalignment="center", 
		color="white" if cm[i, j] > tresh else "black")
	plt.tight_layout()
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.show()
	
def area(a):
	return (a[2] - a[0]) * (a[3] - a[1])

def intersection_area(a, b):  # returns -1 if rectangles don't intersect
	dx = max(a[0], b[0]) - min(a[2], b[2])
	dy = max(a[1], b[1]) - min(a[3], b[3])
	if (dx<=0) and (dy<=0):
		return dx*dy
	return -1

def sorensen_dice_coefficient(frameX, frameY):
	ra = (frameX[0], frameX[2], frameX[1], frameX[3])
	rb = (frameY[0], frameY[2], frameY[1], frameY[3])
	X = area(ra)
	Y = area(rb)
	return 2 * intersection_area(ra, rb) / (X + Y)