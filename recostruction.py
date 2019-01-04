from skimage import io
import imageSegmentation
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from utils import loadDataset, bcolors
from distance import hausdorff
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager


def find_and_subst(i, base_image, centroidSet, predictes, centroidImages, metric='mse'):

    # find the centroid

    if metric != 'hausdorff':
        centroid_index = predictCentroid(centroidSet, predictes[i][0], metric)

    else:
        centroid_index = predictCentroid(centroidImages, predictes[i][0], metric)

    dim = int(np.sqrt(len(centroidImages[centroid_index])))
    replacer = centroidImages[centroid_index].reshape((dim, dim))
    replacer = replacer * 255
    replace_char(base_image, predictes[i][1], predictes[i][2], replacer)

    return True


def replace_char(base_image, r, c, replacer_image):
    # replace the relative image
    img = imresize(replacer_image, (max(r) - min(r) + 1, max(c) - min(c) + 1))
    base_image[min(r):max(r) + 1, min(c):max(c) + 1] = img


def predictCentroid(centroidSet, to_predict, metric='mse'):

    '''
    Compute the closest centroid given an image and depending on a certain metric
    :param img: the input image as 1-D numpy array
    :param centroidsPath: the path of the folder containing th centroids
    :param metric: the metric used to evaluate similarity ('mae', 'mse', 'hausdorff')
    :return: the centroid predicted as 2-D numpy tensor
    '''

    bestCentroidIndex = 0

    if metric == 'mae':
        from sklearn.metrics import mean_absolute_error
        to_predict = to_predict.reshape((100,))
        tmpDistance = mean_absolute_error(to_predict, centroidSet[bestCentroidIndex])
        for i in range(1, len(centroidSet)):

            if mean_absolute_error(to_predict, centroidSet[i]) < tmpDistance:
                tmpDistance = mean_absolute_error(to_predict, centroidSet[i])
                bestCentroidIndex = i

    elif metric == 'mse':
        from sklearn.metrics import mean_squared_error
        to_predict = to_predict.reshape((100,))
        tmpDistance = mean_squared_error(to_predict, centroidSet[bestCentroidIndex])
        for i in range(1, len(centroidSet)):
            if mean_squared_error(to_predict, centroidSet[i]) < tmpDistance:
                tmpDistance = mean_squared_error(to_predict, centroidSet[i])
                bestCentroidIndex = i

    elif metric == 'hausdorff':
        '''
        Hausdorff distance on the original dimension with respect to the centroid images and char images
        '''
        standard_shape = (int(np.sqrt(len(centroidSet[0]))), int(np.sqrt(len(centroidSet[0]))))

        centroidSet = centroidSet * 255

        # In this case in centroidSet there are the real images (np array gray scale on 255)  of the centroids (not encoded)
        # In to predict there is the character to be replaced (not encoded) in matrix form
        tmpDistance = hausdorff(to_predict, centroidSet[bestCentroidIndex].reshape(standard_shape))
        for i in range(1, len(centroidSet)):
            if hausdorff(to_predict, centroidSet[i].reshape(standard_shape)) < tmpDistance:
                tmpDistance = hausdorff(to_predict, centroidSet[i].reshape(standard_shape))
                bestCentroidIndex = i

    return bestCentroidIndex


def replace_char(base_image, r, c, replacer_image):

    # replace the relative image
    img = imresize(replacer_image, (max(r) - min(r)+1, max(c) - min(c)+1))
    base_image[min(r):max(r)+1, min(c):max(c)+1] = img


def arg_wrap(arg):
    return find_and_subst(*arg)


def build_new_image(path, k_means_instance, input_dim, encoder, new_name='_new.png'):

    import sys

    print '\nSelect the distance you want for the choice of the best centroid'
    print '\n', ' ' * 3, '0) Mean Absolute Error compute on the encoded representation'
    print ' ' * 3, '1) Mean Squared Error compute on the encoded representation'
    print ' ' * 3, '2) Hausdorff distance on the binarized original images'

    distances = ['mae', 'mse', 'hausdorff']
    distance_type = int(raw_input("\nDEC> "))
    image = io.imread(path, as_grey=True)

    processed = image
    normalized_height = int(np.sqrt(input_dim))
    normalized_width = normalized_height

    processed = imageSegmentation.image_preproc(processed, binary_threshold=threshold_otsu(processed))
    labeled_image, num_features, max_width, max_height, max_label = imageSegmentation.find_connected_components(processed)

    centroidSet, centroidNames = loadDataset('Centroids')

    centroidImages = centroidSet

    # Get the encoded representation of the centroids
    centroidSet = encoder.predict(centroidSet)

    # Here I create a new image all white that will host each replaced char
    base_image = 255*np.ones(image.shape)

    predictes = list()
    for i in range(1, max_label):

        r_s, c_s = np.where(labeled_image == i)
        if len(r_s) > 1 and len(c_s) > 1:

            # get the char to be replaced
            to_replace = image[min(r_s) - 1:max(r_s) + 2, min(c_s) - 1:max(c_s) + 2]

            # resize before passing through the net
            to_replace = imresize(to_replace, (normalized_height, normalized_width))
            to_predict = to_replace

            if distance_type != 2:
                # get the encoded representation of the char
                to_predict = np.array(to_replace.reshape((1, normalized_height * normalized_width)))
                to_predict = to_predict.astype('float32') / 255.
                to_predict = encoder.predict(to_predict)

            predictes.append([to_predict, r_s, c_s])

    pool = ThreadPoolExecutor(8)
    list_futures = list()
    for i in range(1, len(predictes)):
        list_futures.append(pool.submit(find_and_subst, i, base_image, centroidSet, predictes, centroidImages, metric=distances[distance_type]))

    for i in range(len(list_futures)):
        r = list_futures[i].result()
        sys.stdout.write(u'\u001b[1000D' + bcolors.RED + 'Creating: ' + str(ceil(i * 100 / len(list_futures))) + '%')
        sys.stdout.flush()

    sys.stdout.write(bcolors.RESET)
    plt.imsave('temp.png', base_image, cmap=plt.cm.gray)
