import cv2
import os
from PIL import Image

def cut(src_path, dst_path, edge):
	imlist = os.listdir(src_path)
	for name in imlist:
		img = cv2.imread(src_path + name)
		for i in range(len(alphabet)):
			for j in range(len(alphabet[i])):
				slovo = img[(start_y[i] + edge[0]):(start_y[i] + im_h + edge[1]), (start_x + im_step * j + edge[2]):(start_x + im_step * j + im_w + edge[3])]
				cv2.imwrite(dst_path + alphabet[i][j] + '/' + alphabet[i][j] + '-' + name, slovo)

src_path = 'Ulaz/Formulari/'
dst_path = 'Automatski rezano/'


#OS Fran Frankovic				
start_y = [284, 543, 799, 1055, 1313, 1571, 1829]
start_x = 135
im_h = 193

im_w = 267
im_step = 274

alphabet = [
	['a', 'b', 'v', 'g', 'd'], 
	['e', 'zj', 'dz', 'z', '(i)'], 
	['i', 'dj', 'k', 'l', 'm'], 
	['n', 'o', 'p', 'r', 's'], 
	['t', 'u', 'f', 'h', '(o)'], 
	["(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je'],
    ['ju']
]
edge = [20, -15, 30, -10]

for i in range(1, 4):
	cut(src_path + "Formulari " + str(i) + "/", dst_path + "Formulari/", edge)

#RITEH
start_y = [139, 420, 702, 984, 1265, 1546, 1829]
start_x = 105
im_h = 214

im_w = 300
im_step = 300

alphabet = [
	['a', 'b', 'v', 'g', 'd'], 
	['e', 'zj', 'dz', 'z', '(i)'], 
	['i', 'dj', 'k', 'l', 'm'], 
	['n', 'o', 'p', 'r', 's'], 
	['t', 'u', 'f', 'h', '(o)'], 
	["(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je'],
	['ju']
]
edge = [20, -15, 40, -10]

cut(src_path + "Formulari RITEH/", dst_path + "Formulari/", edge)

#GAM
start_y = [267, 533, 800, 1066, 1329, 1599, 1862]
start_x = 80
im_h = 193

im_w = 267
im_step = 274

alphabet = [
	['a', 'b', 'v', 'g', 'd'], 
	['e', 'zj', 'dz', 'z', '(i)'], 
	['i', 'dj', 'k', 'l', 'm'], 
	['n', 'o', 'p', 'r', 's'], 
	['t', 'u', 'f', 'h', '(o)'], 
	["(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je'],
    ['ju']
]

edge = [20, -15, 40, -10]

cut(src_path + "Formulari GAM/", dst_path + "Formulari/", edge)

#-------------------------------------------
#J i poluglas

edge = [20, -15, 40, -10]

#1
start_y = [284, 543, 799, 1055, 1313, 1571, 1829]
start_x = 135
im_h = 193

im_w = 267
im_step = 274

alphabet = [
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'],
    ['poluglas', 'poluglas', 'poluglas']
]
cut(src_path + "Formulari j i poluglas 1/", dst_path + "Formulari/", edge)

#2
start_y = [284, 543, 799, 1055, 1313, 1571, 1829]
start_x = 135
im_h = 193

im_w = 267
im_step = 274

alphabet = [
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'],
    ['poluglas', 'poluglas', 'poluglas']
]
cut(src_path + "Formulari j i poluglas 2/", dst_path + "Formulari/", edge)

#3
start_y = [284, 543, 799, 1055, 1313, 1571, 1829]
start_x = 135
im_h = 193

im_w = 267
im_step = 274

alphabet = [
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'],
    ['poluglas', 'poluglas', 'poluglas']
]
cut(src_path + "Formulari j i poluglas 3/", dst_path + "Formulari/", edge)

#4
start_y = [315, 581, 845, 1108, 1377, 1642, 1907]
start_x = 130
im_h = 193

im_w = 283
im_step = 274

alphabet = [
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['j', 'j', 'j', 'j', 'j'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'], 
	['poluglas', 'poluglas', 'poluglas', 'poluglas', 'poluglas'],
    ['poluglas', 'poluglas', 'poluglas']
]
cut(src_path + "Formulari j i poluglas 4/", dst_path + "Formulari/", edge)