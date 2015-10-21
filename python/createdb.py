import skimage
from lxml import etree
import os
import glob
from sklearn.cross_validation import train_test_split
import numpy as np
from progress_bar import ProgressBar
import lmdb
import caffe
from skimage import io
import cPickle as pickle
import cv2
from caffe.proto import caffe_pb2


def load_db(db_dir):
    lmdb_env = lmdb.open(db_dir)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    
    X = []
    y = []
    
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        y.append(label)
        X.append(data[0,:,:].ravel())
        
    y = np.array(y).astype(int)
    X = np.array(X)
    
    return X, y


def create_sets(img_dir, train_set_proportion=.6, test_set_proportion=.2, val_set_proportion=.2):

    #val_set_proportion = 1 - train_set_proportion - test_set_proportion

    if os.path.isfile(img_dir+ 'imgs.list'):
        baseimgfilenames = pickle.load(open(img_dir+'imgs.list','rb'))
    else:
        imgfilenames = glob.glob(img_dir + '*.jpg')
	baseimgfilenames = [os.path.basename(f) for f in imgfilenames]                            

    train,val = train_test_split(np.arange(len(baseimgfilenames)),
                                       train_size=train_set_proportion+test_set_proportion,
                                       test_size=val_set_proportion,
                                       random_state=1)  

    train_test_prop = train_set_proportion + test_set_proportion
    train,test = train_test_split(train,                                     
                                  train_size=train_set_proportion/train_test_prop,                                       
                                  test_size=test_set_proportion/train_test_prop,
                                  random_state=1)  

    trainfiles = [baseimgfilenames[i] for i in train]
    valfiles = [baseimgfilenames[i] for i in val]
    testfiles = [baseimgfilenames[i] for i in test]
    
    return trainfiles, valfiles,testfiles
    
    
def write_db(trainfiles, valfiles, testfiles, opts):
    print 'Creating training set'
    train_y, train_X = create_patches_at_center(trainfiles, opts['annotation_dir'], 
                                                    opts['img_dir'], opts['image_dims'][0], 40,grayscale=True)
    if opts['augment-training-data']:
        a_x,a_y = augment(train_X, train_y)
        tolmdb(a_x, a_y, opts['train-dir'])
    else:
        x,y = balance(train_X, train_y)    
        tolmdb(x, y, opts['train-dir'])
        
        
    print '\nCreating validation set'
    val_y, val_X = create_patches_at_center(valfiles, opts['annotation_dir'], 
                                                opts['img_dir'], opts['image_dims'][0], 40,grayscale=True)
    x,y = balance(val_X, val_y)
    tolmdb(x, y, opts['val-dir'])
    
    
    print '\nCreating test set'
    test_y,test_X = create_patches_at_center(testfiles, opts['annotation_dir'], 
                                                 opts['img_dir'], opts['image_dims'][0], 40,grayscale=True)
    x,y = balance(test_X, test_y)
    tolmdb(x, y, opts['test-dir'])


def get_patch_labels_for_single_image(img_filename, image_dir,annotation_dir, size, step):
    '''
    Read the XML annotation files to get the labels of each patch for a 
    given image. The labels are 0 if there is no object in the corresponding
    patch, and 1 if an object is present.
    '''
    annotation_filename = annotation_dir + img_filename[:-3] + 'xml'
    boundingboxes = get_bounding_boxes_for_single_image(annotation_filename)
    img = cv2.imread(image_dir + img_filename)
    height, width, channels = img.shape

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


def get_image_negatives(img, boundingboxes, size, step, grayscale=False):

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
        
    pos= []
    y = (height-(max_y * step))/2
    while y+(size) < height:
            #rows               
            x = (width-(max_x * step))/2

            while (x+(size) < width):
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
                    patch = img[..., top:bottom, left:right]            
                    pos.append(patch)

                x += step
            y += step

            
    return pos

def get_image_positives(img, boundingboxes, size):        
        
    pos = []    
    for bb in boundingboxes:
        cy = (bb[0] + (bb[1]-bb[0])/2)
        cx = (bb[2] + (bb[3]-bb[2])/2)
        patch =  img[..., cx-size/2:cx+size/2,cy-size/2:cy+size/2]
        s= patch.shape
        if s[1]<size or s[2]<size:                        
            continue
        pos.append(patch)
    return pos


def create_patches_at_center(img_basenames, annotation_dir, image_dir, size=50, step=40, grayscale=True, progressbar=True):
    
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
        channels =3

    pos = []
    neg = []
    s = 1
    for img_filename in img_basenames:
        if progressbar:
            pb.step(s)
        s +=1
        annotation_filename = annotation_dir + img_filename[:-3] + 'xml'
        boundingboxes = get_bounding_boxes_for_single_image(annotation_filename)
	colortype = cv2.IMREAD_COLOR
	if grayscale:
		colortype = cv2.IMREAD_GRAYSCALE
        img = cv2.imread(image_dir + img_filename, colortype)  
        if grayscale:
            height,width=img.shape
            img = img.reshape((height, width,channels))
	img = np.rollaxis(img,2)
        image_pos = get_image_positives(img,boundingboxes, size)
        pos.append(image_pos)

        image_neg = get_image_negatives(img, boundingboxes,size,step)
        neg.append(image_neg)
    
    
    pos = [item for sublist in pos for item in sublist]
    
    
    neg = [item for sublist in neg for item in sublist]
    patches = pos+neg    
    
    index = np.arange(len(patches))
    np.random.seed(0)
    np.random.shuffle(index)

    np_patches = np.empty((len(patches),channels,size,size),dtype=np.uint8)
    np_labels = np.empty(len(patches),dtype=int)
    
    
    max_pos=len(pos)
    for i,j in zip(index,xrange(len(index))):        
        if i < max_pos:
            np_patches[j,] = pos[i]
            np_labels[j] = 1
        else:
            np_patches[j,] = neg[i-max_pos]
            np_labels[j] =0
        
    return np_labels,np_patches
    
    
def create_random_patches(img_basenames, annotation_dir, img_dir,size, step, max_imgs=0, grayscale=False): 
        
    if max_imgs == 0:
        N = len(img_basenames)
    else:
        N = max_imgs
    pb = ProgressBar(N)
    
    color_type = 0
    if grayscale:
        channels=1
        
    else:
        channels =3

    num_patches = 0
    instance_list = []

    img = cv2.imread(img_dir +  img_basenames[0])
    height, width, c = img.shape
    
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

    patches = np.empty((N*patches_per_img,channels,size,size),dtype=np.uint8)
        
    labels = []
    num_imgs = 0
    p=0
    for image_idx in range(len(img_basenames)):           
        
        pb.step(image_idx)
        if max_imgs:
            num_imgs += 1
            if num_imgs > max_imgs:
                break
                            
        img_basename = img_basenames[image_idx]
        img_filename = img_dir + img_basename
                   
        # get labels #    
        curr_labels = get_patch_labels_for_single_image(img_basename, img_dir, annotation_dir,
                                                            size, step)
        labels.append(curr_labels)
        #            #
        
        
        colortype = cv2.IMREAD_COLOR
        if grayscale:
                colortype = cv2.IMREAD_GRAYSCALE
        img = cv2.imread( img_filename, colortype)


	if channels == 1:        
            height,width=img.shape
            img = img.reshape((height, width,channels))
        img = np.rollaxis(img,2)
        
        
        y = (height-(max_y * step))/2
        while y+(size) < height:
            #rows               
            x = (width-(max_x * step))/2

            while (x+(size) < width):
                left = x
                right = x+(size)
                top = y
                bottom = y+(size)
                patch = img[..., top:bottom, left:right]
                #io.imsave(patches_dir + img_basename[:-4] + '-' + str(top) + '-' + str(left) + '.jpg' ,patch)
            
                patches[p,] = patch
                p += 1
                
                #patch = patches_dir + img_basename[:-4] + '-' + str(top) + '-' + str(left) + '.jpg' 
                #patch_files.append(patch)                         
                x += step
            y += step
                        
    print '\nSplit each of ' + str(N) + ' images in ' + str(patches_per_img) + \
             ' ' + str(size) + 'x' +  str(size) + ' patches'
    print 'Total number of patches: ' + str(N*patches_per_img)
            
    return  np.array(labels).flatten(), patches
    

def tolmdb(X,y,dbname):
    map_size = X.nbytes * 8
    env = lmdb.open(dbname, map_size=map_size)
    N = X.shape[0]
    pb = ProgressBar(N)    
    db_id=0
    with env.begin(write=True) as txn:
    
      for i in range(N): 
        if i%500==0:
            pb.step(i)
        datum = caffe.io.array_to_datum(X[i],y[i])
        str_id = '{:08}'.format(db_id)
            # txn is a Transaction object
            # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        db_id += 1


# Returns an array with all the positive samples and as many negatives as mult_neg*#pos
def balance(X,y,mult_neg=10):
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


#rotations
def augment(X,y):
    
    shape = X.shape
    num_org=shape[0]
    shape = (shape[0]*4, shape[1], shape[2],shape[3])

    aug_X = np.empty(shape,dtype=np.uint8)
    aug_y =  np.empty(shape[0],dtype=int)
            
    new_patch_order = np.arange(shape[0])
    np.random.shuffle(new_patch_order)
    for i,j in zip(new_patch_order,xrange(shape[0])):
        org_patch = i/4
        rot_n = i%4
        x = np.rollaxis(X[org_patch],0,3 )
        x = np.rot90(x,1)
        rot_X = np.rollaxis(x,2)
 
        aug_X[j,] = (rot_X)
        aug_y[j]=(y[org_patch])
    return aug_X,aug_y
        
    

def get_bounding_boxes_for_single_image(filename):
    '''
    Given an annotation XML filename, get a list of the bounding boxes around
    each object (the ground truth object locations).
    '''
    file_exists = os.path.exists(filename)
    boundingboxes = []

    if (file_exists):
        # Read the bounding boxes from xml annotation
        
        tree = etree.parse(filename)
        r = tree.xpath('//bndbox')
        
        bad = tree.xpath('//status/bad')
        badimage = (bad[0].text=='1')
        
        if badimage: 
            print 'Bad image: ' + annofilename
            exit

        if (len(r) != 0):
            for i in range(len(r)):
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
    
