#import caffe
import numpy as np
import  skimage.filters as filters
#import shapefeatures
#from skimage.feature import peak_local_max
#from skimage.feature import corner_peaks
#from skimage.morphology import watershed
#import skimage.measure as measure
#import skimage.segmentation as segmentation
#import scipy.ndimage as ndimage
#import sklearn
#import sklearn.ensemble
from scipy import misc


def detect(imfile, clf, opts):
    step = opts['detection_step']
    downsample = opts['image_downsample']
    size = opts['patch_size'][0]

    p = predict(clf, imfile, step, size, downsample)

    boxes = get_boxes(imfile, p, step, size, gauss=opts['gauss'], threshold=opts['detection_probability_threshold'] )

    found = non_maximum_suppression(boxes, overlapThresh=opts['detection_overlap_threshold'])
    return found

def predict(classifier, img_filename, step, size, downsample=1):
    img = misc.imread(img_filename)
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
                    patches.append(img[top:bottom:downsample, left:right:downsample,:])
                    x += step
                y += step

    p = np.array(patches)
    p = np.swapaxes(p,1,3)
    p = np.swapaxes(p,2,3)
    predictions = classifier.predict_proba(p)
    
    i=0
    y=0
    while y+(size) < height:
                x = 0
                while (x+(size) < width):
                    left = x
                    right = x+(size)
                    top = y
                    bottom = y+(size)
                    probs[y/step,x/step]=predictions[i,1]
                    i+=1
                    x += step
                y += step

    return probs


def get_boxes(img_filename, probs, step, size, gauss=0,threshold=0.5):

    if gauss != 0:
        probs = filters.gaussian_filter(probs, gauss)
        
    img = misc.imread(img_filename)
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


# Malisiewicz et al.
# Python port by Adrian Rosebrock
def non_maximum_suppression(boxes, overlapThresh=0.5):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  # initialize the list of picked indexes 
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  scores = boxes[:,4]
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the score/probability of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(scores)[::-1]

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))

  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick].astype("int")

'''
def nms_felz(boxes, step, size, lim=0, prob=MAX, pos=MAX, overlapThresh = 0.5, probs=None, probs_area = 90):

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
'''
