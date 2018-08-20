import os
import cv2 as cv
from shutil import copyfile, rmtree
from sklearn.utils import shuffle
from funkcije import prilagodi, noisy

def stratifydata(src, dst, noise_list):
    folders = os.listdir(src)
    for folder in folders:
        imlist = os.listdir(src + folder)
        imlist = shuffle(imlist)

        length = len(imlist)
        trainlength = round(length * 80 / 100)

        trainlist = imlist[:trainlength]
        validationlist = imlist[trainlength:]
        for im in trainlist:
            #copyfile(src + folder + '/' + im, dst + '/train/' + folder + '/' + im)
            img = cv.imread(src + folder + '/' + im,0)
            img = noisy(noise_list, prilagodi(img))
            cv.imwrite(dst + '/train/' + folder + '/' + im, img)
            
        for im in validationlist:
            #copyfile(src + folder + '/' + im, dst + '/validation/' + folder + '/' + im)
            img = cv.imread(src + folder + '/' + im,0)
            img = noisy(noise_list, prilagodi(img))
            cv.imwrite(dst + '/validation/' + folder + '/' + im, img)
			
def resetfolders(main_directory):
	alphabet = ['a', 'b', 'v', 'g', 'd', 'e', 'zj', 'dz', 'z', '(i)', 'i', 'dj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'h', '(o)', "(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je', 'ju', 'j', 'poluglas']
	if os.path.exists(main_directory):
		rmtree(main_directory)
	for i in range(1, 4):
		for letter in alphabet:
			os.makedirs("Raspodjela/data" + str(i) + '/train/' + letter)
			os.makedirs("Raspodjela/data" + str(i) + '/validation/' + letter)

resetfolders("Raspodjela")

stratifydata('Segmentacija/Automatski rezano/Fontovi/', 'Raspodjela/data1', ["s&p"])
stratifydata('Segmentacija/Automatski rezano/Ispravljeni formulari/', 'Raspodjela/data1', ["s&p"])
stratifydata('Segmentacija/Rucno rezano/Papiri/', 'Raspodjela/data1', ["s&p"])
stratifydata('Segmentacija/Rucno rezano/Spovid/', 'Raspodjela/data1', [])
stratifydata('Segmentacija/Rucno rezano/Misal hruacki/', 'Raspodjela/data1', [])

stratifydata('Segmentacija/Automatski rezano/Fontovi/', 'Raspodjela/data2', ["s&p"])
stratifydata('Segmentacija/Rucno rezano/Spovid/', 'Raspodjela/data2', [])
stratifydata('Segmentacija/Rucno rezano/Misal hruacki/', 'Raspodjela/data2', [])

stratifydata('Segmentacija/Automatski rezano/Ispravljeni formulari/', 'Raspodjela/data3', ["s&p"])
stratifydata('Segmentacija/Rucno rezano/Papiri/', 'Raspodjela/data3', ["s&p"])