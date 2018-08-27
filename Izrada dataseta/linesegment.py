import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import bisect
from copy import deepcopy

from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
from sklearn.utils import shuffle
import shutil

def prilagodi(img):
    BLUE = [255,0,0]
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
    img = cv2.copyMakeBorder(img, lijevo, desno, gore, dolje, cv2.BORDER_CONSTANT, value=BLUE)
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
    return img

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
	#c = cv2.adaptiveThreshold(c,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
	c = cv2.adaptiveThreshold(c,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	#c = cv2.GaussianBlur(c,(5,5),0)
	#ret, c = cv2.threshold(c,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	c = cv2.bitwise_not(c)
	backup = deepcopy(c)
	"""
	cv2.imshow("Slika", c)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	h, w = c.shape
	add_slice = []
	last = 0
	for i in range(w):
		if c[h//2, i] > 0:
			flood(c, h//2, i)
			add_slice.append(backup[:, last:i])
			last = i
			"""cv2.imshow("Slika", c)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			"""
	add_slice.append(backup[:, last:])
	return add_slice

def charseg(row_img):
	sol = []
	h, w = row_img.shape
	backup = deepcopy(row_img)
	row_img = cv2.GaussianBlur(row_img,(5,5),0)
	ret, row_img = cv2.threshold(row_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	row_img = cv2.bitwise_not(row_img)

	kernel = np.ones((2, 2),np.uint8)
	row_img = cv2.erode(row_img,kernel,iterations = 2)
	arrimg = np.array(row_img)
	
	hist_col = arrimg.sum(axis=0)
	plt.hist(hist_col, bins=30)
	plt.ylabel('Pikseli s tintom');
	plt.show()
	bef = 0
	left = 0
	right = 0
	for i in range(1, len(hist_col)): # od 1 da left ne ode izvan polja
		if hist_col[i] == 0 and bef > 0:
			right = i
			sol.append(backup[:, left:right])
			#cv2.line(backup, (right, 0), (right , h), 255, 1)
			#cv2.line(backup, (left, 0), (left , h), 1, 1)
		if hist_col[i] > 0 and bef == 0:
			left = i-1
		bef = hist_col[i]
	"""	
	cv2.imshow("Slika", row_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	cv2.imshow("Slika", backup)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print(hist_col)
	"""
	return sol

def place(li, lines, hist, tol, hi, lo):
	if li < lo or li > hi:
		return -1
	li = np.argmin(hist_row[max(li-tol, lo):min(li + tol, hi)]) + li-tol
	for l in lines:
		if abs(l - li) < tol:
			return -1
		if li == l:
			return -1
	return li
	

original = cv2.imread('str.png',0)
img = cv2.GaussianBlur(original,(5,5),0)
ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img = cv2.bitwise_not(img)

kernel = np.ones((3, 3),np.uint8)
img = cv2.erode(img,kernel,iterations = 2)
arrimg = np.array(img)
hist_row = arrimg.sum(axis=1)

"""
plt.hist(hist_row, bins=30)
plt.ylabel('Pikseli s tintom');
plt.show()

print(hist_row)
cv2.imshow("Slika", arrimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

h, w = original.shape
original1 = deepcopy(original)
lines = []
bef = 0
for i in range(len(hist_row)):
	if hist_row[i] == 0 and bef > 0:
		lines.append(i)
	bef = hist_row[i]

avg = h // (len(lines))
#print("Avg:", avg)
lines = [0]
bef = 0
for i in range(len(hist_row)):
	if hist_row[i] == 0 and bef > 0:
		if lines[-1] + avg//3 <= i:
			lines.append(i)
			cv2.line(original, (0, i), (w, i), 1, 1)
	bef = hist_row[i]

lineThickness = 1
prom = True
while (prom):
	prom = False
	for i in range(len(lines)):
		#prava linija
		li = lines[i]
		#print("Linija", i + 1, " -> ", li)
		cv2.line(original, (0, li), (w, li), 1, lineThickness)
		
		#linija iznad sadasnje		
		ind = place(li - avg, lines, hist_row, avg//3, h, 0)
		if ind != -1:
			lines.append(ind)
			lines.sort()
			cv2.line(original, (0, ind), (w, ind), 255, lineThickness)
			prom = True
			break
		#linija ispod sadasnje
		ind = place(li + avg, lines, hist_row, avg//3, h, 0)
		if ind != -1:
			lines.append(ind)
			lines.sort()
			cv2.line(original, (0, ind), (w, ind), 255, lineThickness)
			prom = True
			break

	cv2.imshow("Slika", original)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

chars = []
for i in range(0, 3):
	if lines[i-1] >= lines[i]:
		print("dalje")
		continue
	seg = charseg(original1[lines[i-1]:lines[i], :]) #nakon segmentacije po histogramu
	for c in seg:
		sl = slices(c)
		for s in sl:
			hh, ww = s.shape
			if hh == 0 or ww == 0:
				continue
			"""
			cv2.imshow("Slika", s)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			"""
			chars.append(cv2.bitwise_not(s))
			
#ucitavanje modela
json_file = open('model-data1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#ucitavanje tezina za model
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model-data1.h5")

#kompajlanje modela
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
img_rows, img_cols = 50, 50
num_classes = 33

X = []

imarray = [np.array(prilagodi(im)).flatten() for im in chars]

X += imarray

X = np.array(X)

train_samples = len(X)

if K.image_data_format() == 'channels_first':
    X = X.reshape(train_samples, 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(train_samples, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X = X.astype('float32')

X /= 255

output = loaded_model.predict(X, verbose=0)

azbuka = ['a', 'b', 'v', 'g', 'd', 'e', 'zj', 'dz', 'z', '(i)', 'i', 'dj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'h', '(o)', "(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je', 'ju' ,'j', 'poluglas']
azbuka.sort()			
			
tekst = ""
for i in range(len(output)):
	indeks = output[i].argmax()
	#print(indeks, azbuka[indeks], output[i])

	fig = plt.figure()

	fig.add_subplot(211)
	plt.title('Slovo ' + azbuka[indeks])
	plt.set_cmap('gray')
	plt.imshow(chars[i])

	fig.add_subplot(212)
	plt.plot(output[i])
	plt.title(azbuka[indeks])
	plt.ylabel('vjerojatnost')
	plt.xlabel('slovo')
	plt.show()

	plt.show() 
	tekst += azbuka[indeks] + ' '
	
print(tekst)