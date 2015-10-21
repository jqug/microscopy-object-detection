import numpy as np
import numpy.random as random
from createdb import *
import collections
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from sklearn.metrics import precision_recall_curve, auc
import createdb as cdb

def evaluate_model(detections, image_files, boundingboxes, assigned=None, overlapThreshold=0.5, margin = 0,shape=(0,0)):
    """
    Computes the precision-recall curve of a model.

    Parameters
    ----------
    detections: ndarray
        List of detections including boundingboxes and 
        scores (x1,y1,x2,y2,score).

    image_files: ndarray
        List of filenames of the detections

    boundingboxes: dic
        Dictionary containing the ground truth detections
        of each image_files

    assigned: dic, optional
        Dictionary containing the number of times that
        each ground truth detection is include. Used only
        for bootstrapping

    overlapThreshold: float
        Minimum overlapping threshold to considerer as 
        detection as true positive.

    Returns
    -------
    rec: list
        List of recall values at each score level

    prec: list
        List of precision values at each score level

    p: list
        List of score levels

    """
    if assigned == None:
        assigned = {}
        for bb in boundingboxes.keys() :
            assigned[bb] = np.ones(len(boundingboxes[bb]))    

    
    tp = []
    fp = []
    num_pos = 0
    if margin == 0:
        for k in assigned.keys():
            num_pos += sum(assigned[k])
    else:
        for k in boundingboxes.keys():
            for bb in boundingboxes[k]:
                if  (bb[0] > margin and bb[1] < (shape[1]-margin) and bb[2] > margin and bb[3] < ( shape[0]-margin)):
                    num_pos += 1
    if detections == []:
        return [],[],[]

    
    x1 = detections[:,0]
    y1 = detections[:,1]
    x2 = detections[:,2]
    y2 = detections[:,3]
    p  = detections[:,4].copy()

  
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
     
    idx = np.argsort(-p)
    dp = 0
    for i in idx:
            
            key = image_files[i]
            if margin>0 and (x1[i] < margin or x2[i] > (-margin + shape[1]) or y1[i]< margin or y2[i] > (-margin + shape[0])):
                continue
                

            if boundingboxes[key] == [] or boundingboxes[key].shape[0] == 0 :
                by1 = by2 = bx1 = bx2 = np.array([])
            else:      
                bx1 = boundingboxes[key][:,0]
                bx2 = boundingboxes[key][:,1]            
                by1 = boundingboxes[key][:,2]
                by2 = boundingboxes[key][:,3]



            max_overlap = -1
            max_index = 0
               
                
            for b in range(len(boundingboxes[key])):
                if assigned[key][b] == 0:
                    continue
                    
                if margin>0 and (bx1[b] < margin or bx2[b] > (-margin + shape[1]) or by1[b]< margin or by2[b] > (-margin + shape[0])):                                    
                    continue

                xx1 = max(x1[i], bx1[b])
                yy1 = max(y1[i], by1[b])
                xx2 = min(x2[i], bx2[b])
                yy2 = min(y2[i], by2[b])

                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                areabb = (bx2[b]-bx1[b]) * (by2[b]-by1[b])
                
                
                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / (area[i]+ areabb - float(w*h))
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_index = b
                    
                    

            if max_overlap > overlapThreshold:
                    assigned[key][max_index] -= 1
                    tp.append(1)
                    fp.append(0)
            else:                                    
                    tp.append(0)
                    fp.append(1)
                
    tp = np.cumsum(tp, dtype=np.float32)
    fp = np.cumsum(fp, dtype=np.float32)
        
    rec = tp / num_pos
    pre = tp / (tp+fp)
    
    p.sort()
    return rec, pre, p




def evaluate_detection(detections, annotation_dir, testfiles, threshold=0.5,margin=0,shape=(0,0)):

    boundingboxes = {}
    for key in testfiles:
            annotation_filename = annotation_dir + key[:-3] + 'xml'
            boundingboxes[key] = cdb.get_bounding_boxes_for_single_image(annotation_filename)        
   
    boxes = []
    image_files = []
    

    for key in detections.keys():
    
        img_filename = key
        boxes.append(detections[key])
        image_files.append(  [key for i in xrange(len(detections[key]))]     )
        
    boxes = [j for i in boxes for j in i]
    
    if boxes == []:
        print 'No detections'
        return [],[],[]

    boxes = np.vstack(boxes)
    image_files = np.hstack(image_files)

    return evaluate_model(boxes, image_files, boundingboxes,overlapThreshold=threshold,margin=margin,shape=shape)        


        
def compute_auc(rec, prec):
    """
    Computes the Average Precision 
    (area under the AUC of a PR curve)
    
    Parameters
    ----------
    rec : list
        Recall values
    prec : list
        Precision values
    
    Returns
    -------
    area : float
        average precision
    """

    if len(rec) == 0:
        return 0
    mrec=np.hstack([0,rec,rec[-1]])
    mpre=np.hstack([0,prec,0])

    for i in range(len(mpre)):
        bi = (len(mpre) - i - 2)
        mpre[bi] = max(mpre[bi], mpre[bi+1])



    i=np.where(mrec[1:] != mrec[0:-1])[0]+1;

    area =sum((mrec[i]-mrec[i-1])*mpre[i]);
    return area


def bootstrap(models, annotation_dir, test_set,  iters=10,overlapThreshold=0.5):
    """
    Bootstraps a set of models and computes the AP for each 
    iteration.
    
    Parameters
    ----------
    models : dic
        Contains the detections of each model
    annotation_dir : string
        Path to the annotation files
    test_set : list
        Filenames of the images in the test set
    iters : int, optional
        Number of bootstrapping iteratins, 10 by default
    
    Returns
    -------
    aps: ndarray
        APs of each model for each iteration
    """
    boundingboxes = {}
    
    aps = np.zeros((len(models),iters))

    for f in test_set:                             
        boundingboxes[f] = cdb.get_bounding_boxes_for_single_image(annotation_dir+f[:-3] + 'xml')        
    for bootstrap_iter in range(iters):
        
        selected_imgs = random.randint(0, len(boundingboxes.keys())-1,len(boundingboxes.keys())-1)

        for m_id, model in enumerate( models.keys() ) : 
            selected_bbs = {}
            assigned = {}
            
            for i in selected_imgs:     
                
                if test_set[i] not in selected_bbs:
                    selected_bbs[test_set[i]] = boundingboxes[test_set[i]]
                    assigned[test_set[i]] = np.zeros(len(boundingboxes[test_set[i]]))     
                    
                assigned[test_set[i]] += np.ones(len(boundingboxes[test_set[i]]))        
            
            selected_dets = []
            image_files = []
            
            for i in selected_imgs:                                                                    
                if test_set[i] in models[model]:
                    selected_dets.append(models[model][test_set[i]])
                    image_files.append([test_set[i] for j in xrange(len(models[model][test_set[i]]))])
            selected_dets = [j for i in selected_dets for j in i]
                    
            selected_dets = np.vstack(selected_dets)
            image_files = np.hstack(image_files)

            rec, prec, p = evaluate_model(selected_dets, image_files, selected_bbs, assigned,overlapThreshold=overlapThreshold)
            aps[m_id, bootstrap_iter] = compute_auc(rec,prec)

        
    return aps    


def evaluate_bootstrap(aps_bootstrap, model_aps):
    """
    Print a report of bootstrapping results, including
    the models ranking.
    
    Parameters
    ----------
    aps_bootstrap : ndarray
        APs returned by bootstrap()
    model_aps : dic
        APs of each model without bootstrapping.
        
    Return
    -------
    aps: list
        statistics computed for each model
    keys: list
        name of each statistic
    """

    stats = []
    model_names = model_aps.keys()
    
    for i in range(aps_bootstrap.shape[0]) :
        aps_05 = np.percentile(aps_bootstrap[i,:] ,5)
        aps_50 = np.percentile(aps_bootstrap[i,:] ,50)
        aps_95 = np.percentile(aps_bootstrap[i,:] ,95)
        ap = model_aps.values()[i]
        stats_this = [model_names[i], ap, aps_05,aps_50,aps_95,[]]
        for j in  range(aps_bootstrap.shape[0]) :
            if j==i:
                continue
            diffs = aps_bootstrap[i,:] - aps_bootstrap[j,:]
            
            qb = np.percentile(diffs,5)
            qt = np.percentile(diffs,95)
            

            if (qb <= 0 and qt >= 0):
                stats_this[5].append(model_names[j])
                    
        stats.append(stats_this)

    stats = sorted(stats, key=(lambda x:x[3]), reverse=True)
    for i,model in enumerate(stats):
        print '-', model[0], ':'
        print '\tAP: ', model[1]
        print '\t0.05: ', model[2]
        print '\tMedian: ', model[3]
        print '\t0.95: ', model[4]
        min_rank = i
        num_eq = len(model[5])
        for eq in model[5]:
            for j,model2 in enumerate(stats) :
                if model2[0] == eq and j<i:
                    min_rank = j                            

        
        if num_eq == 0:
            model[5] = str(min_rank+1) 
        else:
            model[5] = str(min_rank+1) + '-' + str(min_rank+num_eq+1)

        print '\tRank: ', model[5]
        
    return stats,['Model','AP','0.05','Median', '0.95', 'Rank']



def show_APs(annotation_dir, testfiles, models,min_overlap=0.15):
    
    current_palette = sns.color_palette()
    x = current_palette[4]
    current_palette[4] = current_palette[3]
    current_palette[3] = x

    results = collections.OrderedDict(sorted(models.items()))

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,10)


    patch_colors = ['royalblue','mediumseagreen', 'crimson', 'purple', 'gold','chocolate']
    patch_colors = current_palette
    patches = []

    i=0
    for k in results.keys():    
            rec, prec, p = evaluate_detection(results[k],annotation_dir,testfiles,min_overlap)        
            area = compute_auc(rec,prec)
            print k, ', AP = ', area
            patches.append(mpatches.Patch(color=patch_colors[i], label= k + ' (' + "{:0.3f}".format(area) + ')' ))
            plt.plot(np.hstack([0,rec,rec[-1]]),
                np.hstack([0,prec,0]) , color=patch_colors[i])

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            plt.ylim([0.0, 1.0])
            plt.xlim([0.0, 1.0])
            plt.title("Overlap threshold: 0.15 (25%)")
            i += 1
            if i == len(current_palette):
                i = 0
    ax1.legend(handles=patches, loc='upper right')
    
def show_aucs(labels,probs ):
    plt.rcParams['figure.figsize'] = (10, 4.5)
    sns.set_style("darkgrid")

    fpr,tpr, th = roc_curve(labels,probs)
    area = roc_auc_score(labels,probs)
    plt.subplot(1,2,1)
    plt.plot(fpr,tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('ROC: AUC=%0.4f' % area)

    precision, recall, thresholds = precision_recall_curve(labels,probs)
    area = auc(recall, precision)    
    plt.subplot(1,2,2)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC=%0.4f' % area)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

def show_prec_loss(steps,accuracy,loss ):
    plt.rcParams['figure.figsize'] = (10, 10)
    sns.set_style("darkgrid")
    plt.subplot(2,1,1)
    plt.plot(steps,accuracy)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,1))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    plt.subplot(2,1,2)
    plt.plot(steps,loss)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,y2))
    plt.title('Loss/iters')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.tight_layout()
    
def show_probs_labels(probs, labels  ):
    
    sns.set_style("dark")

    blue_patch = mpatches.Patch(color='royalblue', label='Negatives')
    green_patch = mpatches.Patch(color='mediumseagreen', label='Positives')


    pos_probs = probs.take(np.where(labels==1)).squeeze()
    neg_probs = probs.take(np.where(labels==0)).squeeze()

    fig, ax1 = plt.subplots()
    ax1.legend(handles=[blue_patch, green_patch], loc='upper center')
    ax1.set_xlabel("Probability")

    plt.rcParams['figure.figsize'] = (10,5)

    bins = [x/10.0 for x in range (11)] 
    ax1.hist(neg_probs,alpha=0.5,bins=bins)

    for tl in ax1.get_yticklabels():
        tl.set_color('blue')

    ax2 = ax1.twinx()
    bins = [x/10.0+0.02 for x in range (11)] 
    bins[10] = 1
    ax2.hist(pos_probs,alpha=0.5,bins=bins,color='green')

    for tl in ax2.get_yticklabels():
        tl.set_color('green',)
    



# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
# From caffe
def vis_square(data, padsize=1, padval=0,interp=None):
    
    vmin = data.min()
    vmax = data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    sns.set_style("white")

    if len(data.shape)>2:
        plt.imshow(data[...,...,0],cmap = plt.get_cmap('gray'), interpolation=interp,vmin=vmin,vmax=vmax)
    else:
        plt.imshow(data,cmap = plt.get_cmap('gray'), interpolation=interp,vmin=vmin,vmax=vmax)

def find_PR_thresholds(rec, prec, p):
    fig = plt.figure()
    fig.set_size_inches(10,5)

    plt.plot(p,rec)
    plt.plot(p,prec)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])

    plt.xlabel('Probability threshold')
    plt.ylabel('Precision/Recall')
    
    points = []
    min_val = 1
    min_index = 0
    for i, v in enumerate(rec):
        if abs(prec[i] - rec[i]) < min_val:
            min_val = abs(rec[i] - prec[i])
            min_index = i        
    print 'ERR: at p = ' + "{:0.2f}".format(p[min_index]) + ' precision = recall = ' + "{:0.2f}".format(rec[min_index] )
    points.append(p[min_index])

    max_val = 0
    max_index = 0

    for i, v in enumerate(p):
        prei = prec[i]    
        reci = rec[i]                
        hmean = 2* (prei*reci)* 1.0/(prei + reci)

        if hmean > max_val:
            max_val = hmean
            max_index = i        
    print 'F1: '+ "{:0.4f}".format(max_val ) + ' at p = ' + "{:0.4f}".format(p[max_index]) \
          + '. Precision: '  + "{:0.4f}".format(prec[max_index] ) + ', recall: ' \
          + "{:0.4f}".format(rec[max_index] )
    points.append(p[max_index])
            
        
    min_val = 1
    min_index = 0
    for i, v in enumerate(p):
        reci = rec[i]            
        if abs(reci-0.9) < min_val:
            min_val = abs(reci-0.9) 
            min_index = i        
                
    print '90% Recall: + at p = ' + "{:0.2f}".format(p[min_index]) \
          + '. Precision: '  + "{:0.2f}".format(prec[min_index] ) + ', recall: ' \
          + "{:0.2f}".format(rec[min_index] )       
    points.append(p[min_index])
    return points


def per_image(found, annotation_dir, img_dir, prob=0.5, threshold=0.5):

    
    num_pos = 0 
    
    boundingboxes = {}
    
    assigned = {}
    
    if len(found.keys()) == 0:
        print 'No detections'
        return [],[],[]
    
    per_image_stats = []
    for key in found.keys(): 
        img = cv2.imread(img_dir + key, cv2.IMREAD_GRAYSCALE)
        probs = np.histogram (img.flatten(), bins=250,density=True,range=(0,255) )[0]

        lprobs = np.log(probs)
        lprobs[np.isneginf(lprobs)] = 0
        
        entropy = -sum(probs * lprobs )
        tp = []
        fp = []
        num_pos
        annotation_filename = annotation_dir + key[:-3] + 'xml'
        boundingboxes[key] = cdb.get_bounding_boxes_for_single_image(annotation_filename)        
        num_pos = len(boundingboxes[key])
        assigned[key] = np.zeros(len(boundingboxes[key]))        

        boxes = found[key]
        
        
        img = img.flatten()

        contrast = img.std()

        
        
        if len(boxes) == 0:
            prec = 0
            rec = 0
            if num_pos == 0:
                prec = 1
                rec = 1
            
            per_image_stats.append(  (num_pos, 0, 0, prec, rec ,entropy,contrast)   )
           
            continue

        boxes = np.vstack(boxes)

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        p  = boxes[:,4].copy()

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
     
        idx = np.argsort(-p)
        dp = 0
        for i in idx:

            if p[i] < prob:
                break
                
            if boundingboxes[key] == [] or boundingboxes[key].shape[0] == 0 :
                by1 = by2 = bx1 = bx2 = np.array([])
            else:      
                bx1 = boundingboxes[key][:,0]
                bx2 = boundingboxes[key][:,1]            
                by1 = boundingboxes[key][:,2]
                by2 = boundingboxes[key][:,3]



            max_overlap = -1
            max_index = 0
               
                
            for b in range(len(boundingboxes[key])):
                if assigned[key][b] != 0:
                    continue

                xx1 = max(x1[i], bx1[b])
                yy1 = max(y1[i], by1[b])
                xx2 = min(x2[i], bx2[b])
                yy2 = min(y2[i], by2[b])

                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                areabb = (bx2[b]-bx1[b]) * (by2[b]-by1[b])
                
                
                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / (area[i]+ areabb)
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_index = b
                    
            if max_overlap > threshold:
                    assigned[key][max_index] = 1
                    tp.append(1)
                    fp.append(0)
            else:                                    
                    tp.append(0)
                    fp.append(1)
                    
        tp = np.sum(tp)
        fp = np.sum(fp)
        
        rec = tp * 1.0/num_pos
        pre = tp * 1.0/(tp+fp) 
        if tp+fp == 0:
                pre = 1
        if num_pos == 0:
            rec = 1
        per_image_stats.append(  (num_pos, tp, fp, pre, rec,entropy,contrast)   )
            
           
    return np.array(per_image_stats),found.keys()


def display_per_image(per_image_stats, xaxis=0, xlabel='Number of parasites',xeqy=False):
    current_palette = sns.color_palette()
    fig,ax = plt.subplots()
    fig.set_size_inches(15,10)


    plt.subplot(2,3,1)
    x = per_image_stats[:,xaxis]
    y = per_image_stats[:,2] +  per_image_stats[:,1]
    plt.scatter(x,y,marker='x',alpha=1)    
    yy = []
    xx = []
    for i in sorted(set(x[3:-3])):
            yy.append(y[np.where(x==i)].mean())
            xx.append(i)
    xp = np.linspace(min(xx[3:-3]), max(xx[3:-3]), 80)    
    p = np.poly1d(np.polyfit(xx,yy, 2))
    plt.plot(xp,p(xp),'red')
    plt.xlabel(xlabel)
    plt.ylabel('Number of detections')

    plt.subplot(2,3,2)
    x = per_image_stats[:,xaxis]
    y = per_image_stats[:,1]
    plt.scatter(x,y,marker='x',alpha=1)
    yy = []
    xx = []
    for i in sorted(set(x[3:-3])):
            yy.append(y[np.where(x==i)].mean())
            xx.append(i)
    xp = np.linspace(min(xx[3:-3]), max(xx[3:-3]), 80)    
    p = np.poly1d(np.polyfit(xx,yy, 2))
    plt.plot(xp,p(xp),'red')
    plt.xlabel(xlabel)
    plt.ylabel('True positives')

    plt.subplot(2,3,3)
    x = per_image_stats[:,xaxis]
    y = per_image_stats[:,2]
    plt.scatter(x,y,marker='x',alpha=1)
    yy = []
    xx = []
    for i in sorted(set(x[3:-3])):
            yy.append(y[np.where(x==i)].mean())
            xx.append(i)
    xp = np.linspace(min(xx[3:-3]), max(xx[3:-3]), 80)    
    p = np.poly1d(np.polyfit(xx,yy, 2))
    plt.plot(xp,p(xp),'red')
    plt.xlabel(xlabel)
    plt.ylabel('False positives')

    plt.subplot(2,3,4)
    x = per_image_stats[:,xaxis]
    y = per_image_stats[:,3]
    plt.scatter(x,y,marker='x',alpha=1)
    yy = []
    xx = []
    for i in sorted(set(x[3:-3])):
            yy.append(y[np.where(x==i)].mean())
            xx.append(i)
    xp = np.linspace(min(xx[3:-3]), max(xx[3:-3]), 80)    
    p = np.poly1d(np.polyfit(xx,yy, 2))
    plt.plot(xp,p(xp),'red')
    plt.xlabel(xlabel)
    plt.ylabel('Precision')

    plt.subplot(2,3,5)
    x = per_image_stats[:,xaxis]
    y = per_image_stats[:,4]
    plt.scatter(x,y,marker='x',alpha=1)
    yy = []
    xx = []
    for i in sorted(set(x[3:-3])):
            yy.append(y[np.where(x==i)].mean())
            xx.append(i)
    xp = np.linspace(min(xx[3:-3]), max(xx[3:-3]), 80)    
    p = np.poly1d(np.polyfit(xx,yy, 2))
    plt.plot(xp,p(xp),'red')
    plt.xlabel(xlabel)
    plt.ylabel('Recall')





def show_detections(listfiles, annotation_dir,img_dir, dects,op,probs_dict,STEP,SIZE):
    fig,ax = plt.subplots()
    fig.set_size_inches(15,20)    

    num_imgs = len(listfiles)
    for i in range(num_imgs):

        k = listfiles[i]

        annotation_filename = annotation_dir+ k[:-3] + 'xml'
        boundingboxes = cdb.get_bounding_boxes_for_single_image(annotation_filename)            

        plt.subplot(num_imgs,2,i*2 + 1)
        img = io.imread(img_dir + k )

        plt.axis('off')

        for bb in boundingboxes:
            cx1 = bb[0] 
            cx2 = bb[1] 
            cy1 = bb[2] 
            cy2 = bb[3] 
            cv2.circle(img, ((cx1+cx2)/2, (cy1+cy2)/2), 20, (255,50,50,255), 3)
        for dd in dects[k]:
            if dd[4] > op:
                cx1 = int(dd[0] )
                cx2 = int(dd[2])
                cy1 = int(dd[1] )
                cy2 = int(dd[3])
                cv2.rectangle(img, (cx1, cy1), (cx2,cy2), (50,255,50,255), 5)

        plt.imshow(img)

        plt.subplot(num_imgs,2,i*2 + 2)    
        prob =  np.uint8(  plt.cm.cubehelix ( probs_dict[k].copy() ) * 255 )
        plt.axis('off')        
        for bb in boundingboxes:
            cx1 = bb[0] / STEP - SIZE/2/STEP
            cx2 = bb[1] / STEP - SIZE/2/STEP
            cy1 = bb[2] / STEP - SIZE/2/STEP
            cy2 = bb[3] / STEP - SIZE/2/STEP
            cv2.circle(prob, ((cx1+cx2)/2, (cy1+cy2)/2), 4, (255,50,50,255), 1)

        for dd in dects[k]:
            if dd[4] > op:
                cx1 = int(dd[0] / STEP - SIZE/2/STEP)
                cx2 = int(dd[2] / STEP - SIZE/2/STEP)
                cy1 = int(dd[1] / STEP - SIZE/2/STEP)
                cy2 = int(dd[3] / STEP - SIZE/2/STEP)
                cv2.rectangle(prob, (cx1, cy1), (cx2,cy2), (50,255,50,255), 1)

        plt.imshow(prob[:-SIZE/STEP,:-SIZE/STEP])
