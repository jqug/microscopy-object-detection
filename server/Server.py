
# coding: utf-8

# In[ ]:

import socket
import sys
import numpy as np
import cv2

s = socket.socket()
s.bind(("localhost",3001)) # remember to change the port and a server address appropriately
s.listen(10)
i=1
while True:
    sc, address = s.accept()
    print address
    f = open("tranmit.jpg",'wb') #open in binary; to be later renamed to nomenclature specific to android phone
    l = 1
    while(l):
        l = sc.recv(1024)
        while (l):
            f.write(l)
            l = sc.recv(1024)
        f.close()
        img = cv2.imread("tranmit.jpg", cv2.IMREAD_GRAYSCALE)
        shape = np.shape(img)[1]
        sc.send(str(shape)) # Would like to send back a response, for now the recieved image dimensions

    sc.close()

s.close()


# In[ ]:




# In[ ]:



