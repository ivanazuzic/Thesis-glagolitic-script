import cv2
import numpy as np
import glob

def create(name, font, img):
    cv2.imwrite('Automatski rezano/Fontovi/' + name + '/' + name + '-' + font + '.png', img)

def extract_letters(img, font, chars):
    index = 0
    imgsize = img.shape
    h = imgsize[0]
    w = imgsize[1]
    left = 0
    right = 0
    intext = 0
    for j in range(w):
        white = 0
        for i in range(h):
            if img[i, j] == 255:
                white = white + 1
        if white == h:
            if intext == 1:
                intext = 0
                right = j 
                letter = img[:, left:right]
                left = j
                if (index < len(chars)):
                    create(chars[index], font, letter)
                    index = index + 1
        else:
            if intext == 0 :
                left = j
                intext = 1

def extract_rows(name):

    img = cv2.imread(name,0)

    # global thresholding
    ret,th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    sh = th.shape
    height = sh[0]
    width = sh[1]
    top = 0
    bottom = 0
    intext = 0

    file = open(name.split('.')[0] + ".txt", "r")

    for i in range(height):
        white = 0
        for j in range(width):
            px = th[i, j]
            if px == 255:
                white = white + 1
        if white == width:
            if intext == 1: 
                intext = 0 
                bottom = i
                row = th[top:bottom, :]
                font = name.split('.')[0]
                font = font.split('\\')[1]
                extract_letters(row, font, file.readline().split())
        else:
            if intext == 0:
                intext = 1
                top = i

    file.close()

for filename in glob.glob('Ulaz/Fontovi/*.png'):
	extract_rows(filename)
