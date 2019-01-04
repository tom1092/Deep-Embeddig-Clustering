from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import imageSegmentation as imS
import clustering as cl
from utils import loadDataset, bcolors
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"



class DEC_model:
    def __init__(self, dims, init='glorot_uniform'):
        '''
        Create the autoencoder model
        :param dims: array which represents the number of units in each layer
        :param init: define the way to set the initial random weights of Keras layers
        '''
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1 #number of hidden layers

        self.autoencoder, self.encoder = self.createAutoencoder(self.dims, init=init)
        self.decoder = self.createDecoder(self.dims, self.autoencoder)

        self.printModels(self.autoencoder, self.encoder, self.decoder)


    def createAutoencoder(self, dims, act='relu', init='glorot_uniform'):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of autoencoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = int(len(dims)/2)
        # input
        x = Input(shape=(dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(n_stacks - 1):
            h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

        # hidden layer
        h = Dense(dims[n_stacks], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

        y = h

        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

        # output
        y = Dense(dims[0], kernel_initializer=init, name='decoder_0', activation='sigmoid')(y)

        return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

    def save_model(self):
        self.autoencoder.save_weights('my_model_weights.h5')

    def load_model(self):
        self.autoencoder.load_weights('my_model_weights.h5')


    def createDecoder(self, dims, autoencoder):
        '''
        Create and return the decoder model
        :param dims: list of number of units in each layer of autoencoder
        :param autoencoder: the keras autoencoder model
        :return: the keras decoder model
        '''

        decStartIndex = int(len(dims)/2)
        dec_input = Input(shape=(dims[decStartIndex],), name='dec_input')
        deco = autoencoder.layers[decStartIndex+1](dec_input)
        for i in range((decStartIndex+2), len(dims)):
            deco = autoencoder.layers[i](deco)
        decoder = Model(dec_input, deco)

        return decoder

    def printModels(self, autoencoder, encoder, decoder):
        '''
        Print the summaries of autoencoder, encoder and decoder models
        '''

        print 'AutoEncoder'
        self.autoencoder.summary()

        print 'Encoder'
        self.encoder.summary()

        print 'Decoder'
        self.decoder.summary()

    def fit(self, optimizer, loss, metric, trainSet, epochs, batchSize):
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        self.autoencoder.fit(trainSet, trainSet, epochs=epochs, batch_size=batchSize, shuffle=True, validation_split=0.02)

    def createClusters(self, testSet, km):

        featureSet = self.encoder.predict(testSet)
        labels, centroids = cl.fit_k_means(featureSet, km)

        if not os.path.exists('Clusters'):
            os.makedirs('Clusters')
        for i in range(K):
            folderClusters = 'Clusters/'+str(i)
            if not os.path.exists(folderClusters):
                os.makedirs(folderClusters)

        decoded_imgs = self.decoder.predict(featureSet)
        from scipy.misc import imsave
        import sys
        from math import ceil
        for i in range(len(labels)):
            strSave = 'Clusters/' + str(labels[i]) + '/' + str(i) + '.png'
            dim = int(np.sqrt(len(decoded_imgs[i])))
            imsave(strSave, decoded_imgs[i].reshape((dim, dim)))
            sys.stdout.write(u'\u001b[1000D'+bcolors.RED+'Saving clusters: '+str(ceil(i*100/len(labels)))+'%')
            sys.stdout.flush()
        if not os.path.exists('Centroids'):
            os.makedirs('Centroids')
            centroids_imgs = self.decoder.predict(centroids)
            for i in range(len(centroids)):
                strSave = 'Centroids/' + str(i) + '.png'
                dim = int(np.sqrt(len(centroids_imgs[i])))
                imsave(strSave, centroids_imgs[i].reshape((dim, dim)))
        sys.stdout.write(bcolors.RESET)
        print


if __name__ == "__main__":
    km = None
    dataset = None
    dims = [2601, 500, 500, 2000, 100, 2000, 500, 500, 2601]
    dec = DEC_model(dims)
    while True:

        print bcolors.CYAN+'\n Scanned image recostruction tool\nwith Keras implementation of Deep Embedding Clustering algorithm\n'+bcolors.RESET
        print '\n Select from the menu:\n'

        print ' '*3, '0) Exit'
        print ' '*3, '1) Image preprocessing: character segmentation'
        print ' '*3, '2) Model fitting '
        print ' '*3, '3) Deep embedding clustering'
        print ' '*3, '4) Image recostruction'
        print ' '*3, '5) Accuracy using OCR and Edit Distance'

        cmdString = int(raw_input("\nDEC> "))

        # Exit
        if cmdString == 0:
            import sys
            sys.exit(0)

        # Image preprocessing and segmentation
        elif cmdString == 1:
            imgPath = str(raw_input('\nInsert the path of the folder images: '))
            imS.doSegmentation(imgPath)

        # Model Fitting
        elif cmdString == 2:
            datasetPath = str(raw_input('Insert the path of dataset: '))
            dataset, datasetfileNames = loadDataset(datasetPath)
            epochs = int(raw_input('Insert number of epochs: '))
            batchSize = int(raw_input('Insert the batch size: '))
            dec.fit('adam', 'mean_squared_error', 'mae', dataset, epochs, batchSize)
            dec.save_model()

        # Clustering
        elif cmdString == 3:
            dec.load_model()
            testSetPath = str(raw_input('Insert the path of the test set: '))
            testSet, testSetFileNames = loadDataset(testSetPath)

            print '\n', ' ' * 3, '0) KMeans'

            clustType = int(raw_input("\nDEC> "))

            if clustType == 0:
                K = int(raw_input('Insert the number of clusters: '))
                km = cl.create_K_means(K)
                dec.createClusters(testSet, km)

        # Recostruction
        elif cmdString == 4:

            scanned_img_path = str(raw_input('Insert the path of image you want to process: '))
            import recostruction
            recostruction.build_new_image(scanned_img_path, km, dec.input_dim, dec.encoder)

        # Evaluate
        elif cmdString == 5:
            scanned_img_path = str(raw_input('Insert the path of the base image: '))
            import evaluate
            result = evaluate.evaluate(scanned_img_path, 'temp.png')
            print result

        else:
            print 'Wrong input'



