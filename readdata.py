import skimage
from lxml import etree
import os
import glob
from sklearn.cross_validation import train_test_split
import numpy as np
from progress_bar import ProgressBar
from skimage import io
from scipy import misc

def create_sets(img_dir, train_set_proportion=.6, test_set_proportion=.2, val_set_proportion=.2):
    '''Split a list of image files up into training, testing and validation sets.'''

    imgfilenames = glob.glob(img_dir + '*.jpg')
    baseimgfilenames = [os.path.basename(f) for f in imgfilenames]

    if train_set_proportion + test_set_proportion < 1:
        train,val = train_test_split(np.arange(len(baseimgfilenames)),
                                           train_size=train_set_proportion+test_set_proportion,
                                           test_size=val_set_proportion,
                                           random_state=1) 
    else:
        train = np.arange(len(baseimgfilenames))
        val = []

    train_test_prop = train_set_proportion + test_set_proportion
    train,test = train_test_split(train,
                                  train_size=train_set_proportion/train_test_prop,
                                  test_size=test_set_proportion/train_test_prop,
                                  random_state=1)

    trainfiles = [baseimgfilenames[i] for i in train]
    testfiles = [baseimgfilenames[i] for i in test]
    valfiles = [baseimgfilenames[i] for i in val]

    return trainfiles, valfiles,testfiles

def get_patch_labels_for_single_image(img_filename, image_dir,annotation_dir, size, step,width, height, objectclass=None):
    '''
    Read the XML annotation files to get the labels of each patch for a
    given image. The labels are 0 if there is no object in the corresponding
    patch, and 1 if an object is present.
    '''
    annotation_filename = annotation_dir + img_filename[:-3] + 'xml'
    boundingboxes = get_bounding_boxes_for_single_image(annotation_filename, objectclass=objectclass)

    # Scan through patch locations in the image
    labels = []
    y = (height-(height/step)*step)/2
    while y+(size) < height:
        #rows
        x = (width-(width/step)*step)/2
        while (x+(size) < width):
            objecthere=0
            for bb in boundingboxes:
                margin = 0
                xmin = bb[0] + margin
                xmax = bb[1] - margin
                ymin = bb[2] + margin
                ymax = bb[3] - margin

                cx = x + size/2
                cy = y + size/2

                if (cx>xmin and cx<xmax and cy>ymin and cy<ymax):
                    objecthere = 1
                    break

            # Output the details for this patch
            labels.append(objecthere)
            x+=step
        y += step

    return np.array(labels)

#http://codereview.stackexchange.com/questions/31352/overlapping-rectangles
def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    return range_overlap(r1[0], r1[1], r2[0], r2[1]) and range_overlap(r1[2], r1[3], r2[2], r2[3])

def get_image_negatives(img, boundingboxes, size, step, grayscale=False, downsample=1, discard_rate=0.9):
    '''Negative-labelled patches, taken at random from any part of the image
    not overlapping an annotated bounding box.

    Since there are typically many potential negative patches in each image, only
    the proprtion 1-discard_rate  of negative patches are stored.'''

    c,height, width = img.shape

    patches_per_img = 0

    #lazy way to count how many patches we can take
    max_y=0
    while max_y+(size) < height:
        max_x = 0
        while max_x+(size) < width:
            patches_per_img += 1
            max_x += step
        max_y += step
    max_x /= step
    max_y /= step

    neg = []
    y = (height-(max_y * step))/2
    while y+(size) < height:
            #rows
            x = (width-(max_x * step))/2

            while (x+(size) < width):
                if np.random.rand()>discard_rate:
                    left = x
                    right = x+(size)
                    top = y
                    bottom = y+(size)

                    is_pos=False
                    for bb in boundingboxes:
                        if overlap([left,right,top,bottom], bb):
                            is_pos=True
                            break

                    if not is_pos:
                        patch = img[:, top:bottom:downsample, left:right:downsample]
                        neg.append(patch.copy())  # without copy seems to leak memory

                x += step
            y += step

    return neg

def get_image_positives(img, boundingboxes, size, downsample=1):
    '''Positive-labelled patches, centred on annotated bounding boxes.'''
    pos = []
    for bb in boundingboxes:
        cy = (bb[0] + (bb[1]-bb[0])/2)
        cx = (bb[2] + (bb[3]-bb[2])/2)
        patch =  img[..., cx-size/2:cx+size/2,cy-size/2:cy+size/2]
        s= patch.shape
        if s[1]<size or s[2]<size:
            continue
        patch = patch[:,::downsample,::downsample]
        pos.append(patch.copy())
    return pos


def create_patches(img_basenames, annotation_dir, image_dir, size, step, grayscale=True, progressbar=True, downsample=1, objectclass=None, negative_discard_rate=.9):
    '''Extract a set of image patches with labels, from the supplied list of
    annotated images. Positive-labelled patches are extracted centered on the
    annotated bounding box; negative-labelled patches are extracted at random
    from any part of the image which does not overlap an annotated bounding box.'''
    if progressbar:
        pb = ProgressBar(len(img_basenames))

    if not annotation_dir[-1] == os.path.sep:
        annotation_dir = annotation_dir + os.path.sep

    if not image_dir[-1] == os.path.sep:
        image_dir = image_dir + os.path.sep

    color_type = 0

    if grayscale:
        channels=1

    else:
        channels=3

    pos = []
    neg = []
    s = 1
    for img_filename in img_basenames:
        if progressbar:
            pb.step(s)
        s +=1
        annotation_filename = annotation_dir + img_filename[:-3] + 'xml'
        boundingboxes = get_bounding_boxes_for_single_image(annotation_filename, objectclass)
        #colortype = cv2.IMREAD_COLOR

        #img = cv2.imread(image_dir + img_filename, colortype)
        img = misc.imread(image_dir + img_filename)
        height,width,channels=img.shape
        img = img.reshape((height, width,channels))
        img = np.rollaxis(img,2)
        image_pos = get_image_positives(img,boundingboxes,size,downsample=downsample)
        pos.append(image_pos)

        image_neg = get_image_negatives(img,boundingboxes,size,step,downsample=downsample,discard_rate=negative_discard_rate)
        neg.append(image_neg)

    pos = [item for sublist in pos for item in sublist]
    neg = [item for sublist in neg for item in sublist]
    patches = pos+neg

    index = np.arange(len(patches))
    np.random.seed(0)
    np.random.shuffle(index)

    np_patches = np.empty((len(patches),channels,size/downsample,size/downsample),dtype=np.uint8)
    np_labels = np.empty(len(patches),dtype=int)

    max_pos=len(pos)
    for i,j in zip(index,xrange(len(index))):
        if i < max_pos:
            np_patches[j,] = pos[i]
            np_labels[j] = 1
        else:
            np_patches[j,] = neg[i-max_pos]
            np_labels[j] = 0

    np_labels = np_labels.astype(np.uint8)
    return np_labels,np_patches

def balance(X,y,mult_neg=10):
    '''Returns an array with all the positive samples and as many negatives as
    mult_neg*npos'''
    np.random.seed(0)
    neg = np.where(y==0)[0]
    neg_count = len(neg)
    pos = np.where(y==1)[0]
    pos_count = len(pos)
    np.random.shuffle(neg,)
    neg = neg[0:pos_count*mult_neg]
    index = np.concatenate((pos, neg))
    np.random.shuffle(index)
    y = y.take(index)
    X = X.take(index,axis=0)
    return X,y

def augment(X,y):
    '''Create rotated and flipped versions of all patches.'''

    shape = X.shape
    num_org=shape[0]
    shape = (shape[0]*8, shape[1], shape[2], shape[3])

    aug_X = np.empty(shape,dtype=np.uint8)
    aug_y =  np.empty(shape[0],dtype=int)

    new_patch_order = np.arange(shape[0])
    np.random.shuffle(new_patch_order)

    for i,j in zip(new_patch_order,xrange(shape[0])):
        orig_patch = i/8
        rot_n = i%4
        do_flip = i%8>3
        x = np.rollaxis(X[orig_patch],0,3 )
        if do_flip:
            x = np.flipud(x)
        x = np.rot90(x,rot_n)
        rot_X = np.rollaxis(x,2)

        aug_X[j,] = (rot_X)
        aug_y[j]=(y[orig_patch])

    aug_y = aug_y.astype('uint8')

    return aug_X,aug_y


def augment_positives(X,y):
    '''Create rotated and flipped versions of only the positive-labelled
    patches.'''
    pos_indices = np.where(y)[0]
    neg_indices = np.where(y==0)[0]

    aug_X_pos, aug_y_pos = augment(X[pos_indices,], y[pos_indices])
    aug_X = np.vstack((aug_X_pos, X[neg_indices,]))
    aug_y = np.hstack((aug_y_pos, y[neg_indices]))

    new_order = np.random.permutation(aug_y.shape[0])
    aug_X = aug_X[new_order,]
    aug_y = aug_y[new_order]

    aug_y = aug_y.astype('uint8')

    return aug_X, aug_y


def get_bounding_boxes_for_single_image(filename, objectclass=None):
    '''
    Given an annotation XML filename, get a list of the bounding boxes around
    each object (the ground truth object locations).
    '''
    annofile = filename[:-3] + 'xml'
    file_exists = os.path.exists(filename)
    boundingboxes = []

    if (file_exists):
        # Read the bounding boxes from xml annotation
        tree = etree.parse(filename)
        r = tree.xpath('//bndbox')

        if (len(r) != 0):
            for i in range(len(r)):
                if (objectclass==None) or (objectclass in r[i].getparent().xpath('label')[0].text.lower()):
                    xmin = round(float(r[i].xpath('xmin')[0].text))
                    xmin = max(xmin,1)
                    xmax = round(float(r[i].xpath('xmax')[0].text))
                    ymin = round(float(r[i].xpath('ymin')[0].text))
                    ymin = max(ymin,1)
                    ymax = round(float(r[i].xpath('ymax')[0].text))
                    xmin, xmax, ymin, ymax = int(xmin),int(xmax),int(ymin),int(ymax)

                    boundingboxes.append((xmin,xmax,ymin,ymax))

    if len(boundingboxes) == 0:
        return np.array([])

    return np.vstack(boundingboxes)
