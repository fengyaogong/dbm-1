import numpy as np

class FirstLayerRBM:
    def __init__(self, data, nHidden, maxEpoch):
        self.nHidden = nHidden
        self.maxEpoch = maxEpoch
        self.CD = 1
        self.data = data

        #e - epsilon - leraning rate
        self.eWeights = 0.05
        self.eVisibleBiases = 0.05
        self.eHiddenBiases = 0.05
        
        #TODO: what are this
        self.weightCost = 0.001
        self.initialMomentum = 0.5
        self.finalMomentum = 0.9
        
        #Number of: examples (in one batch), features and batches in the current data set
        (self.nBatches, self.nExamples, self.nFeatures) = data.shape
        
        #In which epoch of the training algorithm are is the program
        self.currentEpoch = 0

        #Visible -> Hidden weights
        self.wVisHid = 0.001 * np.random.randn(self.nFeatures, self.nHidden)
        
        #Biases for hidden and visible layer
        self.bHidden = np.zeros((1, self.nHidden))
        self.bVisible = np.zeros((1, self.nFeatures))
        
        #Probabilities in the positive and negative phase for the hidden layer
        self.positiveHiddenProbs = np.zeros((self.nExamples, self.nHidden))
        self.negativeHiddenProbs = np.zeros((self.nExamples, self.nHidden))
        
        #Producst in positive and negative phase
        self.positiveProducts = np.zeros((self.nFeatures, self.nHidden))
        self.negativeProducts = np.zeros((self.nFeatures, self.nHidden))
        
        #TODO:
        self.visibleHiddenIncrement = np.zeros((self.nFeatures, self.nHidden))

        #TODO: Check if this are increments for visible and hidden biases
        self.bHiddenInc = np.zeros((1, self.nHidden))
        self.bVisibleInc = np.zeros((1, self.nFeatures))
        
        #Positive probabilities for every batch,example and hidden unit
        self.batchPositiveHiddenProbs = np.zeros((self.nBatches, self.nExamples, self.nHidden))
        
    def train(self):
        for epoch in range(self.currentEpoch, self.maxEpoch):
            self.currentEpoch = epoch
            print "Current epoch: {}, max epoch: {}".format(self.currentEpoch, self.maxEpoch)

            #TODO: what is this
            errsum = 0

            for batch in range(self.nBatches):
                #Bias for every example in one batch
                bVisibleOne = np.tile(self.bVisible, (self.nExamples, 1))
                bHiddenOne = np.tile(2 * self.bHidden, (self.nExamples, 1))
                
         ########################## Start of the positive phase ################################       
                #Convert the input data from one batch to 1/0 
                data = self.data[batch]
                data = (data > np.random.rand(self.nExamples, self.nFeatures)).astype(int)

                self.positiveHiddenProbs = 1 / (1 + np.exp(np.dot(-data, (2 * self.wVisHid)) - bHiddenOne))
                self.batchPositiveHiddenProbs[batch] = self.positiveHiddenProbs
                
                self.positiveProducts = np.dot(np.transpose(data), self.positiveHiddenProbs)
                positiveHiddenActive = np.sum(self.positiveHiddenProbs, 0)
                positiveVisibleActive = np.sum(data, 0)
        ########################## End of positive phase ########################################

        ########################## Start of the negative phase ##################################
                positiveHiddenStates = (self.positiveHiddenProbs > np.random.rand(self.nExamples, self.nHidden)).astype(int)
                #Reconstruction of data
                negativeData = 1 / (1 + np.exp(np.dot(-positiveHiddenStates, np.transpose(self.wVisHid)) - bVisibleOne))
                negativeData = (negativeData > np.random.rand(self.nExamples, self.nFeatures)).astype(int)
                self.negativeHiddenProbs = 1 / (1 + np.exp(np.dot(-negativeData, (2 * self.wVisHid)) - bHiddenOne))

                self.negativeProducts = np.dot(np.transpose(negativeData), self.negativeHiddenProbs)
                negativeHiddenActive = np.sum(self.negativeHiddenProbs, 0)
                negativeVisibleActive = np.sum(negativeData, 0)

        ########################## End of negative phase #######################################
                err = np.sum( np.square(data - negativeData) )
                errsum = errsum + err

                if epoch > 5:
                    momentum = self.finalMomentum
                else:
                    momentum = self.initialMomentum

        ########################## Update weights and biases ###################################
                self.visibleHiddenIncrement = momentum * self.visibleHiddenIncrement + \
                        self.eWeights * ( (self.positiveProducts - self.negativeProducts) / self.nExamples - \
                        self.weightCost * self.wVisHid)
                self.bVisibleInc = momentum * self.bVisibleInc + (self.eVisibleBiases / self.nExamples) * \
                        (positiveVisibleActive - negativeVisibleActive)
                self.bHiddenInc = momentum * self.bHiddenInc + (self.eHiddenBiases / self.nExamples) * \
                        (positiveHiddenActive - negativeHiddenActive)
                
                self.wVisHid = self.wVisHid + self.visibleHiddenIncrement
                self.bVisible = self.bVisible + self.bVisibleInc
                self.bHidden = self.bHidden + self.bHiddenInc
        ########################## End of weights and biases updates ###########################

            print "End of epoch: {}, errsum: {}".format(epoch, errsum)



