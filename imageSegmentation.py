from skimage import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from scipy import ndimage
from scipy.ndimage import measurements
from skimage.measure import regionprops
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.patches as mpatches
import numpy as np
from math import ceil
import os
import glob


def get_interline(I):

    C = I[:, int(ceil(len(I[1, :]) / 4))+1]
    zerosCount = 0
    found = 0
    v = list()
    for i in range(C.shape[0]):
        if C[i] == 0:
            found = 1
            zerosCount = zerosCount + 1

        else:

            if found == 1:
                v.append(zerosCount)
                found = 0
            zerosCount = 0

    v = np.sort(v)
    to_delete = int(0.20 * len(v))
    v = v[to_delete:(len(v) - to_delete)]
    mean = np.sum(v) / len(v)
    return mean


def cc_labeling(A, i, j, value, ref):

    M = np.zeros(A.shape)
    while A[i-1, j] == ref:
       i = i - 1

    k = i
    while A[k, j] == ref:
        A[k, j] = value
        k = k + 1
    i = i - 1
    while i <= k:

       if A[i, j-1] == ref:
           A = cc_labeling(A, i, j-1, value, ref)

       if A[i, j+1] == ref:
           A = cc_labeling(A, i, j+1, value, ref)
       i = i +1
    M = A
    return M


def adjust_label(I):
    M = I
    interline = get_interline(M)
    print "Average interline: "+str(interline)+" pixels"
    # Call the adjustment on each column for the detection of near classes that probably are the same character (e.g "i")

    for j in range(M.shape[1]):
        zeroValues = 0
        A = M[:, j]

        # Check if in the column there's at least a black pixel
        if np.count_nonzero(A) > 0:
            ref, pos = get_first_element(A) #Get the label and the position of the first black pixel in that column
            for i in range(pos, len(A)):

                if A[i] == 0:
                   zeroValues = zeroValues +1
                else:
                    # If they are near more than 25% of interline, than the two objects are in the same class
                    if zeroValues < 0.25*interline and A[i] != ref:
                        M = cc_labeling(M, i, j, ref, A[i])

                    else:
                        zeroValues = 0
                        ref = A[i]
    return M


def get_first_element(A):

    i = 0
    while i < len(A) and A[i] == 0:
        i = i + 1
    ref = A[i]
    pos = i
    return ref, pos


def invert(A):
    for i in range(A.shape[0]):
        A[i] = [1 if A[i, j] == 0 else 0 for j in range(A.shape[1])]
    return A


def open_image(path_to_file="", as_grey=True):

    if path_to_file != "":
        img = io.imread(path_to_file, as_grey=as_grey)
        plt.title("IMAGE WITH NOISE")
        plt.imshow(img, cmap="gray")
        plt.show()
        return img


def image_preproc(img, blur_radius=None, binary_threshold=None, min_structure=np.ones((5, 6))):

    ndimage.binary_opening(img, structure=min_structure)
    if blur_radius is not None:
        # apply a BLUR filter
        img = ndimage.gaussian_filter(img, blur_radius)

    if binary_threshold is not None:
        # apply binarization
        img = binarize(img, threshold=binary_threshold, copy=True)

    return img


def find_connected_components(img):

    # find connected components
    # label image regions we want 8-connected so we define a structure
    s = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    img = invert(img)

    labeled_image, num_features = measurements.label(img, structure=s)
    max_label = num_features
    labeled_image = adjust_label(labeled_image)
    #fig, ax = plt.subplots(figsize=(10, 6))
    img = invert(img)
    #ax.imshow(img, cmap="gray", interpolation='nearest')
    num_features = len(regionprops(labeled_image))
    max_height = 0
    max_width = 0
    count = 1
    for region in regionprops(labeled_image):

        count += 1
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='#88ff4d', linewidth=1)
        #ax.add_patch(rect)
        if max_height < abs(minr-maxr):
            max_height = abs(minr-maxr)

        if max_width < abs(minc-maxc):
            max_width = abs(minc-maxc)

    #ax.set_axis_off()
    #plt.tight_layout()
    #plt.show(block=False)
    return labeled_image, num_features, max_width, max_height, max_label


def save_images(images, Ls, Fs, max_width, max_height, folder_name):

    try:
        os.makedirs(folder_name)
    except OSError:
        if not os.path.isdir(folder_name):
            raise
    for j in range(len(images)):
        for i in range(1, Fs[j]):
            # find the component with label i
            r, c = np.where(Ls[j] == i)
            if len(r):

                # build the relative image
                char_image = images[j][min(r)-1:max(r)+2, min(c)-1:max(c)+2]

                if(max_width >= max_height):
                    resizedImage = imresize(char_image, (max_width, max_width))
                else:
                    resizedImage = imresize(char_image, (max_height, max_height))

                baseFileName = 'Char_'+str(j)+'-'+str(i)+'.png'
                fullFileName = str(folder_name) + '/' + str(baseFileName)
                imsave(fullFileName, resizedImage)


def doSegmentation(images_path):
    images = [io.imread(path, as_grey=True) for path in glob.glob(images_path+'/*.png')]
    Ls = list()
    Fs = list()
    maxW = 0
    maxH = 0
    count = 1
    for img in images:

        print '\nProcessing document: '+str(count)+' of '+str(len(images))
        img = image_preproc(img, binary_threshold=0.40)
        labeled_image, num_features, max_width, max_height, max_label = find_connected_components(img)
        Ls.append(labeled_image)
        Fs.append(max_label)
        maxW = max_width if max_width > maxW else maxW
        maxH = max_height if max_height > maxH else maxH
        count += 1

    #print max_height, max_width
    print '\nSaving all characters....'
    save_images(images, Ls, Fs, maxW, maxH, images_path+'_folder')

