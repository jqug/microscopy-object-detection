import caffe
import numpy as np
import  skimage.filters as filters
import shapefeatures
from skimage.feature import peak_local_max
from skimage.feature import corner_peaks
from skimage.morphology import watershed
import skimage.measure as measure
import skimage.segmentation as segmentation
import scipy.ndimage as ndimage

MAX = 0
MEAN = 1
AREA = 2
NUM = 3

def detect(imfile, net, opts):
    p = network_predict(net, imfile)
    boxes = get_boxes(imfile, probs=p, gauss=opts['gauss'], threshold=opts['threshold'] )
    found = nms_felz(boxes, lim=opts['lim'], prob = opts['prob'], pos = opts['pos'], 
                    overlapThresh=opts['overlapThreshold'], probs=p,probs_area = opts['probs_area'], step = opts['step'])
    return found
    
def network_get_prob(patches, classifier, oversample=True):
    for i in range(len(patches)):
        patches[i] = patches[i].reshape(50,50,1)
  
    return classifier.predict(patches, oversample)

def tree_get_prob(patches, classifier, oversample=True):

    feats = []
    
    featureset = [3,7,11,12,15,17]
    filters = [[11,'>',1000]]
    centiles = [0,25,50,75,100]
    print patches[0].shape

    for patch in patches:
        feats.append(shapefeatures.extract(patch, featureset, filters, centiles))

    p = np.vstack(feats)
    return classifier.predict_proba(p)[:,1]

     
def network_predict(net, img_filename, step=5, size=50, oversample=True, norm=False):
    return    predict(net, network_get_prob, img_filename, step, size, oversample, norm)
    
def tree_predict(classifier, img_filename, step=5, size=50, oversample=True):
    return    predict(classifier, tree_get_prob, img_filename, step, size, oversample)
    
def predict(classifier, predict_func, img_filename, step=5, size=50, oversample=True,norm=False):
    img = caffe.io.load_image(img_filename)
    height, width,channels = img.shape

    probs = np.zeros((img.shape[0]*1.0/step,img.shape[1]*1.0/step))
    patches = []
    
    y=0    
    while y+(size) < height:
                #rows     
                x = 0
                predictions=[]
                while (x+(size) < width):
                    left = x
                    right = x+(size)
                    top = y
                    bottom = y+(size)
		    if norm:
			p = img[top:bottom, left:right,1]
                        m = p.min()
			M = p.max()
			p = (p-m) * 1.0 / (M-m)
                        patches.append(p) 
		    else:
                        patches.append(img[top:bottom, left:right,1]) 
                    x += step
                y += step

    predictions = predict_func(patches, classifier, oversample=oversample ) 
    i=0
    y=0
    while y+(size) < height:
                x = 0
                while (x+(size) < width):
                    left = x
                    right = x+(size)
                    top = y
                    bottom = y+(size)
                    probs[y/step,x/step]=predictions[i][1]
                    i+=1
                    x += step
                y += step

    return probs


def get_boxes(img_filename, probs, gauss=0,threshold=0.5, step=5, size=50):
    
    if gauss != 0:            
        probs = filters.gaussian_filter(probs, gauss)

    
    img = caffe.io.load_image(img_filename)
    height, width,channels = img.shape
    
    boxes=[]
    
    i=0
    y=0
    while y+(size) < height:
                x = 0
                while (x+(size) < width):
                    left = int(x)
                    right = int(x+(size))
                    top = int(y)
                    bottom = int(y+(size))
                    if probs[y/step,x/step] > threshold:
                        boxes.append([left,top,right,bottom,probs[y/step,x/step]])                                          
                    i+=1
                    x += step
                y += step
                
    if len(boxes) == 0:
        return np.array([])

    boxes =  np.vstack(boxes)
    return boxes


def nms_felz(boxes, lim=0, prob=MAX, pos=MAX, overlapThresh = 0.5, size=50, probs=None, probs_area = 40, step = 5 ):
    
    probs_area = int(probs_area / step)
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.array([])
 
    # initialize the list of picked indexes
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    p  = boxes[:,4]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(p)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        suppress = [last]
        
        merged_probs = [p[i]]
        merged_c = []
        merged_c.append( ( (x1[i] + x2[i])/2.0, (y1[i] + y2[i])/2.0 ) )

        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]
 
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
 
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
 
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
 
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
                   
                
                merged_probs.append(p[j])                    
                
                merged_c.append( ( (x1[j] + x2[j])/2.0, (y1[j] + y2[j])/2.0 ) )

        if len(merged_probs) >= lim:
               
            if pos==MEAN:
                    tot_prob = sum(merged_probs)
                    box_center_x = sum(  [  merged_c[i][0] * merged_probs[i]/tot_prob for i in xrange(len(merged_probs))])
                    box_center_y = sum(  [  merged_c[i][1] * merged_probs[i]/tot_prob for i in xrange(len(merged_probs))])
            else:
                box_center_x = (x1[i]+x2[i] ) /2 
                box_center_y = (y1[i]+y2[i] ) /2 

            pr = 0
            if prob == MEAN:
                pr = sum(merged_probs)
                pr *= (1.0/(len(merged_probs)))                            
            elif prob == AREA:
                pr = probs[box_center_y - probs_area/2 : box_center_y + probs_area/2,  box_center_x - probs_area/2 : box_center_x + probs_area/2 ].mean()
            elif prob == NUM:
                pr = sum(merged_probs)
            else:
                pr = p[i]

            pick.append([ box_center_x-size/2, box_center_y-size/2, box_center_x+size/2, box_center_y+size/2, pr])

        idxs = np.delete(idxs, suppress)
    if len(pick)== 0:
        return np.array([])
    # return only the bounding boxes that were picked
    return np.vstack(pick)


def nms_own(probs, minth, minpix, gauss=0,step=5,size=50):

    if gauss != 0:            
        probs_img = filters.gaussian_filter(probs, gauss)
    else:        
        probs_img=probs.copy()

    probs_img[probs_img<minth] = 0
    probs_img[probs_img>=minth] = 1
    probs_img = ndimage.binary_dilation( ndimage.binary_erosion(probs_img)).astype(np.float64)
    probs_img = ndimage.binary_erosion( ndimage.binary_dilation(probs_img)).astype(np.float64)

    distance = ndimage.distance_transform_edt(probs_img)

    local_maxi = peak_local_max(distance, min_distance=3, indices=False,labels=probs_img,exclude_border=False)
    markers = measure.label(local_maxi, connectivity=2)
    labels_ws = watershed(-distance, markers, mask=probs_img>minth)

    found = []

    #plt.imshow(labels_ws)
    for i in range(1,len(labels_ws)+1):
        if  sum( sum(labels_ws==i)) <= minpix:
            labels_ws[labels_ws==i]=0
        else:
            _,y,x = np.where([labels_ws==i])
    
            xm = (x.mean() +step) * step
            ym = (y.mean() +step) * step
            
            p = probs[ y.mean()- size/step/2 : y.mean()+size/step/2, x.mean()-size/step/2:x.mean()+size/step/2 ].flatten().mean()
            
            if p > 0:
                found.append((xm-size/2,ym-size/2,xm +size/2 ,ym+size/2, p))      
                    
    return found

