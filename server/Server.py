
# coding: utf-8

# In[ ]:

import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2
import detectobjects as det
import createdb as cdb
from convnet import ConvNetClassifier

opts = {'img_dir': '/some/where/images',
        'model_dir': '/some/where/models/models/',
        'annotation_dir': '/some/where/annotation',
        'model': '2C-1FC-O',
        'threshold': 0.5, 
        'overlapThreshold': 0.3, 
        'lim': 0, 
        'prob': det.MAX, 
        'pos': det.MAX, 
        'gauss': 1,
        'mean': np.array([162.83]),
        'input_scale': None,
        'raw_scale': 255,
        'image_dims': (50,50),
        'channel_swap': None,
        'probs_area': 40,
        'step': 5
       }

net = ConvNetClassifier(opts)
trainfiles, valfiles, testfiles = cdb.create_sets(opts['img_dir'])

s = socket.socket()
s.bind(('',xxxx) # remember to change the port and a server address appropriately
s.listen(10)
i=1
while True:
    sc, address = s.accept()
    print address
    f = open("transmit.jpg",'wb') #open in binary; to be later renamed to nomenclature specific to android phone
    l = 1
    while(l):
        l = sc.recv(1024)
        while (l):
            f.write(l)
            l = sc.recv(1024)
        f.close()
        img = cv2.imread("transmit.jpg", cv2.IMREAD_GRAYSCALE)
        #imfile = opts['img_dir'] + testfiles[1]
        imfile = '/some/where/tranmit.jpg'
        found = det.detect(imfile, net, opts)
        #shape = np.shape(img)[1]
        sc.send(str(found[1])) # Would like to send back a response, for now the recieved parasite locations

    sc.close()

s.close()
