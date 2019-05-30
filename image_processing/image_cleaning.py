#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:17:34 2019

@author: papadaki
"""

import glob
import os
from PIL import Image
import numpy as np
import subprocess
import json

x = []
y = []
WIDTH = 224
HEIGHT = 224
ASPECTRATIO = 1
img_list = {os.path.basename(x).rsplit('_',1)[0] for x in glob.iglob('goss_images_nonreduced/*')}
img_done = {os.path.basename(x).rsplit('.',1)[0] for x in glob.iglob('images_gossipcop2/*')}

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr


for filename in glob.iglob('images_politifact_raw/*'):
    print(filename)
    img = Image.open(filename)
    width, height = img.size
    if 0.5 <= width/height <= 1.5:
        new_img = img.resize((WIDTH, HEIGHT), resample=1)
    elif width < height:
        new_height = width/ASPECTRATIO
        new_width = width
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        new_img = img.crop((left, top, right, bottom))
        new_img = new_img.resize((WIDTH, HEIGHT), resample=1)
    elif width > height:
        new_height = height
        new_width = np.floor(min(width, ASPECTRATIO*height))
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        new_img = img.crop((left, top, right, bottom))
        new_img = new_img.resize((WIDTH, HEIGHT), resample=1)
    # arr = np.array(new_img)
    # norm_img = Image.fromarray(normalize(arr).astype('uint8'),'RGB')
    outname = 'images_resized/' + os.path.basename(filename)
    # norm_img.save(outname)
    new_img.convert('RGB').save(outname)

"""
plan:  if image is close to same aspect ratio then resize w/ distortion. If 
very different, center crop something with same aspect ratio and then resize to 
right dimensions
goal size:  450x300 (1.5 aspect ratio)
"""

articles = {os.path.basename(x) for x in glob.iglob('images_resized/poli*')}
articles = list(articles)
art_arr = np.array(x for x in articles)


from sklearn.model_selection import train_test_split
xTrain, xTest = train_test_split(articles, test_size = 0.2, random_state = 0)

for art in xTrain:
    search = 'images_resized/' + art + '*'
    images = glob.iglob(search)
    for im in images:
        subprocess.call(['mv', im, 'images_resized/train/'])

for art in xTest:
    search = 'images_resized/' + art + '*'
    images = glob.iglob(search)
    for im in images:
        subprocess.call(['mv', im, 'images_resized/test/'])

labels = {}
poli_fake = [os.path.basename(x) for x in glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/politifact/fake/*')]
poli_true = [os.path.basename(x) for x in glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/politifact/real/*')]

for article in poli_fake:
    labels[article] = {'label': 1}

for article in poli_true:
    labels[article] = {'label': 0}

with open('images_resized/labels.json', 'w') as outfile:
    json.dump(labels, outfile)

with open('images_resized/labels.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    for art in data:
        lab = data[art]['label']
        if lab == 0:
            try:
                search = 'images_resized/train/' + art + '*'
                images = glob.iglob(search)
                for im in images:
                    print(im)
                    subprocess.call(['mv', im, 'images_resized/train/True/'])
            except:
                print('something is wrong')
        elif lab == 1:
            try:
                search = 'images_resized/train/' + art + '*'
                images = glob.iglob(search)
                for im in images:
                    subprocess.call(['mv', im, 'images_resized/train/Fake/'])
            except:
                print('something is wrong again')
        else:
            print('something is wrong yet again')


# normalize images
images = list(glob.iglob('images/*/*/*'))
for image_file in images:
    img = Image.open(image_file)
    arr = np.array(img)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGB')
    new_img.save(image_file)
