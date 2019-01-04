def loadDataset(datasetPath):
    '''
    Load the characters images dataset and return a numpy 2-D tensor of the dataset\n
    :param datasetPath: folder path of the dataset
    :return: the 2-D tensor with shape: (n_sample, n_features) and the sorted list of file names
    '''

    import os
    print('Total char images: ', len(os.listdir(datasetPath)))

    from PIL import Image
    import glob
    import numpy as np
    imagesPath = datasetPath + '/*.png'
    filelist = glob.glob(imagesPath)

    fileNameDataset = []
    for fname in filelist:
        fileNameDataset.append(fname)

    dataset = np.array([np.array(Image.open(fname)) for fname in filelist])
    dataset = dataset.astype('float32') / 255.
    dataset = dataset.reshape((dataset.shape[0], -1))

    print('Shape of dataset tensor: ', dataset.shape)
    return dataset, fileNameDataset


class bcolors:

    PINK = '\033[95m'
    CYAN = u'\u001b[36m'
    BLUE = u'\u001b[34m'
    GREEN = u'\u001b[32m'
    YELLOW = u'\u001b[33m'
    RED = u'\u001b[31m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = u'\u001b[0m'