import numpy as np

class FirstLayerRBM:
    def __init__(self, data, nHidden, maxEpoch, currentEpoch):
        self.nHidden = nHidden
        self.maxEpoch = maxEpoch
        self.CD = 1
        self.data = data
        np.random.seed(555)
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
        self.currentEpoch = currentEpoccurrentEpoch

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
    
    def continueTD(self, epoch, wVisHid, bVisible, bHidden):
        self.currentEpoch = epoch
        self.wVisHid = wVisHid
        self.bVisible = bVisible
        self.bHidden = bHidden

    def saveState(self, path):
        np.save(path + "/currentEpoch0", self.currentEpoch)
        np.save(path + "/wVisHid0", self.wVisHid)
        np.save(path + "/bVisible0", self.bVisible)
        np.save(path + "/bHidden0", self.bHidden)

    def loadState(self, path):
        self.currentEpoch = np.load(path + "/currentEpoch0.npy")
        self.wVisHid = np.load(path + "/wVisHid0.npy")
        self.bVisible = np.load(path + "/bVisible0.npy")
        self.bHidden = np.load(path + "/bHidden0.npy")
 
    def train(self):
        for epoch in range(self.currentEpoch, self.maxEpoch):
            self.currentEpoch = epoch
            #print "Current epoch: {}, max epoch: {}".format(self.currentEpoch, self.maxEpoch)

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


class LastLayerRBM:
    def __init__(self, nHidden, maxEpoch, nLabels, data, targets, nHiddenPre, wVisHidPre, bHiddenPre, bVisiblePre, cd, currentEpoch):
        np.random.seed(666)
        
        self.nHidden = nHidden
        self.maxEpoch = maxEpoch
        self.CD = cd
        
        #The number of steps for CD, in fact this is calculated as 
        # NumberOfSteps = current_epoch / nStepsCD
        self.nStepsCD = 20
        self.data = data
        self.targets = targets

        #e - epsilon - leraning rate
        self.eWeights = 0.05
        self.eVisibleBiases = 0.05
        self.eHiddenBiases = 0.05
        
        #TODO: what are this
        self.weightCost = 0.001
        self.initialMomentum = 0.5
        self.finalMomentum = 0.9
        
        #Previous layer weights and biases
        self.wVisHidPre = wVisHidPre
        self.bHiddenPre = bHiddenPre
        self.bVisiblePre = bVisiblePre

        #Number of: examples (in one batch), features and batches in the current data set
        (self.nBatches, self.nExamples, self.nFeatures) = data.shape
       
        #The hidden layer from previous 'layer' becomes the visible layer of this
        self.nFeaturesPre = self.nFeatures
        self.nFeatures = nHiddenPre
        #In which epoch of the training algorithm are is the program
        self.currentEpoch = currentEpoch

        #Visible -> Hidden weights
        self.wVisHid = 0.01 * np.random.randn(self.nFeatures, self.nHidden)
        
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
        
        self.nLabels = nLabels
        self.labelsHidden = 0.01 * np.random.randn(self.nLabels, self.nHidden)
        self.bLabels = np.zeros((1, self.nLabels))
        self.labelsHiddenInc = np.zeros((self.nLabels, self.nHidden))
        self.bLbalesInc = np.zeros((1, self.nLabels))
    
    def continueTD(self, epoch, wVisHid, bVisible, bHidden, bLabels, wLabHid):
        self.currentEpoch = epoch
        self.wVisHid = wVisHid
        self.bVisible = bVisible
        self.bHidden = bHidden
        self.bLabels = bLabels
        self.labelsHidden = wLabHid


    def saveState(self, path):
        np.save(path + "/currentEpoch1", self.currentEpoch)
        np.save(path + "/wVisHid1", self.wVisHid)
        np.save(path + "/bVisible1", self.bVisible)
        np.save(path + "/bHidden1", self.bHidden)
        np.save(path + "/bLabels1", self.bLabels)
        np.save(path + "/wLabHid1", self.labelsHidden)

    def loadState(self, path):
        self.currentEpoch = np.load(path + "/currentEpoch1.npy")
        self.wVisHid = np.load(path + "/wVisHid1.npy")
        self.bVisible = np.load(path + "/bVisible1.npy")
        self.bHidden = np.load(path + "/bHidden1.npy")
        self.bLabels = np.load(path + "/bLabels1.npy")
        self.labelsHidden = np.load(path + "/wLabHid1.npy")


    def train(self):
        for epoch in range(self.currentEpoch, self.maxEpoch):
            self.currentEpoch = epoch
            
            #print "Current epoch {}/{}".format(epoch, self.maxEpoch)
            self.CD = np.ceil((epoch + 1) / float(self.nStepsCD))
            eWeights = self.eWeights / float(self.CD)
            eVisibleBiases = self.eVisibleBiases / float(self.CD)  
            eHiddenBiases = self.eHiddenBiases / float(self.CD)
            errsum = 0
            for batch in range(self.nBatches):
        ############################ Start of the positive phase ###############################
                dataPre = self.data[batch]
                positiveHiddenProbsPre = 1 / (1 + np.exp(np.dot(-dataPre, 2 * self.wVisHidPre) - np.tile(2 * self.bHiddenPre, (self.nExamples, 1))))
                data = (positiveHiddenProbsPre > np.random.rand(self.nExamples, self.nFeatures)).astype(int)
                #TODO: check are these the real targets
                targets = self.targets[batch]
                
                bHiddenOne = np.tile(self.bHidden, (self.nExamples, 1))
                bVisibleOne = np.tile(2 * self.bVisible, (self.nExamples, 1))
                bLabelsOne = np.tile(self.bLabels, (self.nExamples, 1))

                #TODO: change name of labHid to wLabelsHidden
                self.positiveHiddenProbs = 1 / (1 + np.exp(np.dot(-data, self.wVisHid) - np.dot(targets, self.labelsHidden) - bHiddenOne))
                self.positiveProducts = np.dot( np.transpose(data), self.positiveHiddenProbs )
                positiveProductsLabelsHidden = np.dot( np.transpose(targets), self.positiveHiddenProbs )

                positiveHiddenActive = np.sum(self.positiveHiddenProbs, 0)
                positiveVisibleActive = np.sum(data, 0)
                positiveLabelsActive = np.sum(targets, 0)
        ############################ End of positive phase #####################################
                positiveHiddenProbs_tmp = np.array(self.positiveHiddenProbs)
        ############################ Start of the negative phase ###############################
                for iterCD in range(int(self.CD)):
                    positiveHiddenStates = (positiveHiddenProbs_tmp > np.random.rand(self.nExamples, self.nHidden)).astype(int)
                    totin = np.dot(positiveHiddenStates, np.transpose(self.labelsHidden)) + bLabelsOne 
                    negativeLabelProbs = np.exp(totin)
                    negativeLabelProbs = negativeLabelProbs / (np.dot( np.reshape(np.sum(negativeLabelProbs, 1), (-1,1)), np.ones((1, self.nLabels))))

                    xx = np.cumsum(negativeLabelProbs, 1)
                    xx1 = np.random.rand(self.nExamples,1)
                    negativeLabelStates = negativeLabelProbs * 0
                    for jj in range(self.nExamples):
                        #TODO: check this
                        index = np.min(np.where(xx1[jj] <= xx[jj,:]))
                        negativeLabelStates[jj, index] = 1
                    xxx = np.sum(negativeLabelStates)

                    negativeData = 1 / (1 + np.exp(np.dot(-positiveHiddenStates, np.transpose(2 * self.wVisHid)) - bVisibleOne))
                    negativeData = (negativeData > np.random.rand(self.nExamples, self.nFeatures)).astype(int)
                    positiveHiddenProbs_tmp = 1 / (1 + np.exp(np.dot(-negativeData, self.wVisHid) - np.dot(negativeLabelStates, self.labelsHidden) - 
                        bHiddenOne))

                self.negativeHiddenProbs = np.array(positiveHiddenProbs_tmp)

                self.negativeProducts = np.dot(np.transpose(negativeData), self.negativeHiddenProbs)
                negativeHiddenActive = np.sum(self.negativeHiddenProbs, 0)
                negativeVisibleActive = np.sum(negativeData, 0)
                negativeLabelActive = np.sum(negativeLabelStates, 0)
                negativeProductsLabelHidden = np.dot(np.transpose(negativeLabelStates), self.negativeHiddenProbs)
        ############################ End of negative phase #######################################
                err = np.sum( np.square(data - negativeData) )
                errsum = errsum + err

                if epoch > 5:
                    momentum = self.finalMomentum
                else:
                    momentum = self.initialMomentum
 
        ############################ Weight updates ##############################################
                self.visibleHiddenIncrement = momentum * self.visibleHiddenIncrement + \
                        eWeights * ( (self.positiveProducts - self.negativeProducts) / self.nExamples - \
                        self.weightCost * self.wVisHid)
                self.labelsHiddenInc = momentum * self.labelsHiddenInc + \
                        eWeights * ( (positiveProductsLabelsHidden - negativeProductsLabelHidden) / self.nExamples \
                        - self.weightCost * self.labelsHidden)
                
                self.bVisibleInc = momentum * self.bVisibleInc + (eVisibleBiases / self.nExamples) * \
                        (positiveVisibleActive - negativeVisibleActive)
                self.bHiddenInc = momentum * self.bHiddenInc + (eHiddenBiases / self.nExamples) * \
                        (positiveHiddenActive - negativeHiddenActive)
                self.bLbalesInc = momentum * self.bLbalesInc + (eVisibleBiases / self.nExamples) * \
                        (positiveLabelsActive - negativeLabelActive)

                self.wVisHid = self.wVisHid + self.visibleHiddenIncrement
                self.labelsHidden = self.labelsHidden + self.labelsHiddenInc 
                
                self.bVisible = self.bVisible + self.bVisibleInc
                self.bHidden = self.bHidden + self.bHiddenInc
                self.bLabels = self.bLabels + self.bLbalesInc

            print "End of epoch: {}, errsum: {}".format(epoch, errsum)
