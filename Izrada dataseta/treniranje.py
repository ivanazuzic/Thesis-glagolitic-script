from __future__ import print_function
import os
#da odaberem graficku karticu
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import shutil

#importanje Kerasa i slojeva za arhitekturu1
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

#importanje slojeva za VGG16
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D
#optimizer za VGG16 model
from tensorflow.keras.optimizers import SGD

#importanje alata za augumentaciju podataka
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#importanje dodatnih metrika iz Kerasa
from tensorflow.keras import metrics

#importanje numpya
import numpy as np
#importanje matplotliba za grafove
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import shutil
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import model_from_json
from PIL import Image

#sprema graf za usporedbu dviju funkcija
def save_graph(x, y1, y2, x_label, y_label, y1_label, y2_label, plot_title, dst):
	fig = plt.figure()
	ax = plt.subplot(111)
	line1 = plt.plot(x, y1, 'b', label=y1_label)
	plt.setp(line1, color='r', linewidth=1.0)
	line2 = plt.plot(x, y2, 'b', label=y2_label)
	plt.setp(line2, color='g', linewidth=1.0)
	plt.title(plot_title)
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	plt.legend([y1_label, y2_label], loc='upper left')
	#plt.grid(True)
	fig.savefig(dst)

#funkcija koja vraca VGG16 model
def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=input_shape))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
				  optimizer=tensorflow.keras.optimizers.Adadelta(),
				  metrics=['accuracy'])
	return model

#funkcija koja vraca LeNet model
def lenet():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
				  optimizer=tensorflow.keras.optimizers.Adadelta(),
				  metrics=['accuracy']) #metrics.categorical_accuracy
	return model

#funkcija koja vraca prvi model za kineski jezik
#neke linije su komentirane jer se pokazalo da je originalni model predubok za glagoljicu	
def kineski1():
	model = Sequential()
	model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(160, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(192, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	"""
	model.add(Conv2D(224, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(288, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(320, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(352, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(384, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	"""
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
				  optimizer=tensorflow.keras.optimizers.Adadelta(),
				  metrics=['accuracy']) #metrics.categorical_accuracy
	return model

#funkcija koja vraca drugi model za kineski jezik
#neke linije su komentirane jer se pokazalo da je originalni model predubok za glagoljicu	
def kineski2():
	model = Sequential()
	model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(160, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	"""
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(Conv2D(192, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
	"""
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
				  optimizer=tensorflow.keras.optimizers.Adadelta(),
				  metrics=['accuracy']) #metrics.categorical_accuracy
	return model

#funkcija koja vraca model za arapski jezik
def arapski():
	model = Sequential()
	model.add(Conv2D(72, kernel_size=(6, 6), strides=1, activation='relu', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Conv2D(144, (5, 5), strides=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(192, (4, 4), strides=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(400, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
				  optimizer=tensorflow.keras.optimizers.Adadelta(),
				  metrics=['accuracy']) #metrics.categorical_accuracy
	return model

#funkcija koja trenira model na odredjenom datasetu te sprema rezultat
def treniraj(model, foldername, modelname, data_number, dataset, epochs, batch_size, steps_per_epoch, validation_steps):
	name = foldername + '/' + modelname + '/' + modelname + '-data'+ str(data_number)
	train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=False)

	val_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
			dataset + '/train',
			target_size=(img_rows, img_cols),
			color_mode='grayscale',
			batch_size=batch_size,
			class_mode='categorical',
			shuffle='True')

	validation_generator = val_datagen.flow_from_directory(
			dataset + '/validation',
			target_size=(img_rows, img_cols),
			color_mode='grayscale',
			batch_size=batch_size,
			class_mode='categorical',
			shuffle='True')

	es = tensorflow.keras.callbacks.EarlyStopping(
				monitor='val_loss',
				min_delta=0,
				patience=2,
				verbose=0, 
				mode='auto')
				
	history = model.fit_generator(
			train_generator,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			verbose=1,
			validation_data=validation_generator,
			validation_steps=validation_steps,
			callbacks = [es])

	#spremanje modela
	model_json = model.to_json()
	with open(name + ".json", "w") as json_file:
		json_file.write(model_json)
	#spremanje tezina
	model.save_weights(name + ".h5")

	num_epochs = [i for i in range(1, len(history.history['acc']) + 1)]
	save_graph(num_epochs, history.history['acc'], history.history['val_acc'], 'Epohe', 'Točnost', 'Točnost treniranja', 'Točnost validacije', 'Graf točnosti', 'Rezultati/' + modelname + '/' + 'plot-' + modelname + '-data'+ str(data_number) + '-acc.png')
	save_graph(num_epochs, history.history['loss'], history.history['val_loss'], 'Epohe', 'Gubitak', 'Gubitak treniranja', 'Gubitak validacije', 'Graf gubitka', 'Rezultati/' + modelname + '/' + 'plot-' + modelname + '-data'+ str(data_number) + '-loss.png')

#funkcija koja prikazuje matricu zabune
def cm_saving(cm, classes, dst, normalize='False', title='Matrica zabune'):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(cm), cmap=plt.cm.jet, 
					interpolation='nearest')
	plt.title(title)

	width = len(classes)
	height = len(classes)

	cb = fig.colorbar(res)
	alphabet = classes
	plt.xticks(range(width), alphabet[:width], rotation=90)
	plt.yticks(range(height), alphabet[:height])
	plt.tight_layout(pad=1.5)
	plt.ylabel("Prava klasa")
	plt.xlabel("Predviđena klasa")
	plt.savefig(dst, format='png')
	#plt.show()
	
def validiraj(model, val):	
	azbuka = ['a', 'b', 'v', 'g', 'd', 'e', 'zj', 'dz', 'z', '(i)', 'i', 'dj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'h', '(o)', "(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je', 'ju' ,'j', 'poluglas']
	azbuka.sort()

	#ucitavanje modela
	json_file = open(model + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#ucitavanje tezina
	loaded_model.load_weights(model + '.h5')
	print("Loaded model from disk")

	img_rows, img_cols = 50, 50
	num_classes = 33

	test_labels = []
	rounded_predictions = []
	
	tocno = 0
	krivo = 0
	for slovo in azbuka:
		X = []

		path = val + "/" + slovo + '/'
		imlist = os.listdir(path)

		imarray = [np.array(Image.open(path + im)).flatten() for im in imlist]

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
		print(slovo)
		output = loaded_model.predict(X, verbose=1)
	
		for i in range(len(output)):
			indeks = output[i].argmax()
			test_labels.append(azbuka.index(slovo)) #koja je to klasa
			rounded_predictions.append(indeks) #kako je ispalo
			if azbuka.index(slovo) == indeks:
				tocno += 1
			else:
				krivo += 1

	cm_plot_labels = ['(i)', '(o)', "(š)ć", 'a', 'b', 'c', 'cj', 'd', 'dj', 'dž', 'e', 'f', 'g', 'h', 'i', 'j', 'ja,(i)je', 'ju', 'k', 'l', 'm', 'n', 'o', 'p', 'pol.', 'r', 's', 'š', 't', 'u', 'v', 'z', 'ž']
	cm = confusion_matrix(test_labels, rounded_predictions)
	#cm_plot_labels = [i for i in range(1, 34)]
	cm_saving(cm, cm_plot_labels, 'Rezultati/' + model + val + '.png')
	
	file = open('Results/' + model + val + '.txt','w') 
	acc_by_class = []
	acc = []
	for i in range(len(cm)):
		acc_by_class.append((cm[i][i] / test_labels.count(i),  i) )
		acc.append((cm[i][i] / test_labels.count(i),  i))
		file.write("Slovo " + azbuka[i] + " TP " + str (cm[i][i]) + " Total " +str(test_labels.count(i)) + '\n')
	
	acc_by_class.sort()
	acc_by_class.reverse()
	for a in acc_by_class[:5]:
		file.write(azbuka[a[1]] + ' ' + str(a[0]) + ' ' + str(cm[a[1]]) + '\n')
	file.write('---------------\n')
	file.write('Total accuracy ' + str(accuracy_score(test_labels, rounded_predictions, normalize=True)) + '\n')
	file.write('Total accuracy2 ' + str(tocno / (tocno + krivo)) + '\n')
	file.write('---------------\n')
	for a in acc:
		file.write(azbuka[a[1]] + ' ' + str(a[0]) + '\n')
	file.close()
	
img_rows, img_cols = 50, 50
input_shape = (img_rows, img_cols, 1)
num_classes = 33
d_azbuka = {'(i)': '(i)', '(o)':'(o)', "(sj)c'": "(š)ć", 'a': 'a', 'b': 'b', 'c': 'c', 'cj': 'č', 'd': 'd', 'dj': 'đ', 'dz': 'dž', 'e': 'e', 'f': 'f', 
	'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j', 'ja, (i)je': 'ja,(i)je', 'ju': 'ju', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 
	'poluglas': 'pol.', 'r': 'r', 's': 's', 'sj': 'š', 't': 't', 'u': 'u', 'v': 'v', 'z': 'z', 'zj': 'ž'}

#resetiranje mapa koje nastaju treniranjem
svi_modeli = ["modelLeNet", "modelarapski", "modelkineski1", "modelkineski2", "modelVGG"]

if os.path.exists("Modeli"):
	shutil.rmtree("Modeli")	
os.makedirs("Modeli")

if os.path.exists("Rezultati"):
	shutil.rmtree("Rezultati")	
os.makedirs("Rezultati")

for mod in svi_modeli:
	os.makedirs("Modeli/" + mod)
	os.makedirs("Rezultati/" + mod)

#save_graph([1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [5, 15, 20, 45, 80], 'Epohe', 'Točnost', 'Točnost klasifikacije', 'Točnost validacije', 'Graf točnosti', 'Rezultati/' + 'modelLeNet' + '/' + 'plot-' + 'modelLeNet' + '-data'+ str(1) + '-acc.png')
	
for i in range(1, 4):
	treniraj(lenet(), 'Modeli/', 'modelLeNet', i, 'Raspodjela/data' + str(i), 50, 120, 120, 120)
	treniraj(arapski(), 'Modeli/', 'modelarapski', i, 'Raspodjela/data' + str(i), 50, 120, 120, 120)
	treniraj(kineski1(), 'Modeli/', 'modelkineski1', i, 'Raspodjela/data' + str(i), 50, 120, 120, 120)
	treniraj(kineski2(), 'Modeli/', 'modelkineski2', i, 'Raspodjela/data' + str(i), 50, 120, 120, 120)
	#treniraj(VGG_16(), 'Modeli/', 'modelVGG', i, 'Raspodjela/data' + str(i), 5, 120, 120, 120)