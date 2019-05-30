#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:40:59 2019

@author: papadaki
"""

import json
import glob
import os
import requests
from PIL import Image, ImageFile
from io import BytesIO
from urllib.request import urlopen, Request
import signal

image_formats = ("image/png", "image/jpeg", "image/gif", "image/jpg")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}


def getsizes(uri):
    # get file size *and* image size (None if not known)
    req = Request(uri, headers=headers)
    file = urlopen(req)
    size = file.headers.get("content-length")
    if size: size = int(size)
    p = ImageFile.Parser()
    while 1:
        data = file.read(1024)
        if not data:
            break
        p.feed(data)
        if p.image:
            return size, p.image.size
            break
    file.close()
    return size, None


def handler(signum, frame):
    raise Exception("timed out")


path = "FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake"
poli_fake = [os.path.basename(x) for x in glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/*')]
counter = 0
img_done = {os.path.basename(x).split('.')[0] for x in glob.iglob('images_gossipcop/*')}
signal.signal(signal.SIGALRM, handler)
badurl = 'https://pmcvariety.files.wordpress.com/2019/04/xboxonesad.jpg'

for filename in glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/gossipcop/*/*/news content.json'):
    if counter % 100 == 0:
        print(str(counter), "articles completed")
    dirname = os.path.basename(os.path.dirname(filename))
    if dirname in img_done:
        print('skipping',dirname)
        counter += 1
        continue
    print(dirname)
    with open(filename, 'r') as json_file:
        imidx = 1
        data = json.load(json_file)
        images = data['images']
        w_max = 0
        h_max = 0
        biggest = ""
        for im in images:
            try:
                signal.alarm(15)
                r = requests.head(im)
                if r.headers["content-type"] in image_formats and not im==badurl: # check if the content-type is a image
                    w, h = getsizes(im)[1]
                    if w*h > w_max*h_max:
                        w_max = w
                        h_max = h
                        biggest = im
            except:
                pass
            finally:
                signal.alarm(0)
                
        if biggest != "":
            img_data = requests.get(biggest, headers={'User-Agent': 'firefox'}).content
            print(dirname, 'success!')
            img_name = "images_gossipcop/" + dirname + ".jpg"
            with open(img_name, 'wb') as handler:
                handler.write(img_data)
    counter += 1


labels = {}
poli_fake = [os.path.basename(x) for x in glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/gossipcop/fake/*')]
poli_true = [os.path.basename(x) for x in glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/gossipcop/real/*')]

for article in poli_fake:
    labels[article] = {'label': 1, 'images': []}

for article in poli_true:
    labels[article] = {'label': 0, 'images': []}


for img in glob.iglob('images_gossipcop/*'):
    imgname = os.path.basename(img)
    article = imgname.split('_')[0]
    labels[article]['images'].append(imgname)

with open('gossipcop_labels.json', 'w') as outfile:
    json.dump(labels, outfile)


# from jpegtran import JPEGImage
