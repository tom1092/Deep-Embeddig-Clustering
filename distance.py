import numpy as np


def getBlackPixelCoord(image):

    '''
    Given a gray-scale image (with values in [0,1]) return the coordinates of black pixels
    :param image: the gray-scale image as 2-D numpy tensor with values in range [0,1]
    :return: the coordinates of black pixels as 2-D numpy tensor
    '''

    blackRowPixel, blackColPixel = np.where(image == 0)
    #blackRowPixel, blackColPixel = normalizeCoordSet(blackRowPixel, blackColPixel)

    return blackRowPixel, blackColPixel


def hausdorffBlackPixelDistance(blackCoordImA, blackCoordimB):

    '''
    Calculate the hausdorff distance between imA and imB tensor
    :param imA: 2-D tensor with each element represent the coordinates of black pixels in the first image
    :param imB: 2-D tensor with each element represent the coordinates of black pixels in the second image
    :return: the hausdorff distance between imA and imB
    '''
    from scipy.spatial.distance import directed_hausdorff

    return max(directed_hausdorff(blackCoordImA, blackCoordimB)[0], directed_hausdorff(blackCoordimB, blackCoordImA)[0])


def normalizeCoordSet(blackCoordRow, blackCoordCol): #TODO: da rivedere
    baricenter = np.zeros((2,))
    for i in range(len(blackCoordRow)):
        baricenter = baricenter + [blackCoordRow[i], blackCoordCol[i]]
    baricenter = baricenter / len(blackCoordRow)

    for i in range(len(blackCoordRow)):
        blackCoordRow[i] = blackCoordRow[i] - baricenter[0]
        blackCoordCol[i] = blackCoordCol[i] - baricenter[1]

    scaleFactor = 100/(max(blackCoordRow) - min(blackCoordCol))
    blackCoordRow = blackCoordRow * scaleFactor
    blackCoordCol = blackCoordCol * scaleFactor

    return blackCoordRow, blackCoordCol


def hausdorff(img1, img2):

    from sklearn.preprocessing import binarize
    from skimage.filters import threshold_otsu

    # Avoid binarize of the original ones
    copy1 = binarize(img1, threshold=threshold_otsu(img1))
    copy2 = binarize(img2, threshold=threshold_otsu(img2))

    # Get row and column of black pixels
    black1_row, black1_col = getBlackPixelCoord(copy1)
    black2_row, black2_col = getBlackPixelCoord(copy2)

    # Baricenter normalization
    black1_row, black1_col = normalizeCoordSet(black1_row, black1_col)
    black2_row, black2_col = normalizeCoordSet(black2_row, black2_col)

    black1 = np.vstack((black1_row, black1_col)).T
    black2 = np.vstack((black2_row, black2_col)).T

    return hausdorffBlackPixelDistance(black1, black2)









