import numpy as np
import mf

class DBM:
    def __init__(self, maxEpoch, data, targets, nLabels, nHidden, nPen, wVisHid1, bLabels1, bHidden1, bVisible1, labelsHidden1, wVisHid0, bHidden0, bVisible0, currentEpoch):
        np.random.seed(222)
        self.targets = targets
        self.maxEpoch = maxEpoch
        self.eWeight = 0.001   # Learning rate for weights 
        self.eVisibleBias = 0.001   # Learning rate for biases of visible units 
        self.eHiddenBias = 0.001   # Learning rate for biases of hidden units 
        self.weightCost = 0.0002
        self.initialMomentum = 0.5
        self.finalMomentum = 0.9
        self.data = data

        (self.nBatches, self.nExamples, self.nFeatures) = self.data.shape

        self.nLabels = nLabels
        self.nHidden = nHidden
        self.nPen = nPen        

        self.currentEpoch = currentEpoch
        # Initializing symmetric weights and biases.

        self.wVisHid = 0.001 * np.random.randn(self.nFeatures, self.nHidden);
        self.wHidPen = 0.001 * np.random.randn(self.nHidden, self.nPen);

        self.wLabPen = 0.001 * np.random.randn(self.nLabels, self.nPen);

        self.bHidden = np.zeros((1, self.nHidden));
        self.bVisible  = np.zeros((1, self.nFeatures));
        self.bPen = np.zeros((1, self.nPen));
        self.bLabels = np.zeros((1, self.nLabels));
        
        #Probabilities in the positive and negative phase for the hidden layer
        self.positiveHiddenProbs = np.zeros((self.nExamples, self.nHidden))
        self.negativeHiddenProbs = np.zeros((self.nExamples, self.nHidden))
        self.positivePenProbs = np.zeros((self.nExamples, self.nPen))

        #Producst in positive and negative phase
        self.positiveProducts = np.zeros((self.nFeatures, self.nHidden))
        self.negativeProducts = np.zeros((self.nFeatures, self.nHidden))

        
        self.visibleHiddenIncrement = np.zeros((self.nFeatures, self.nHidden))
        self.hiddenPenIncrement = np.zeros((self.nHidden, self.nPen))
        self.labelsPenIncrement = np.zeros((self.nLabels, self.nPen))

        self.bHiddenInc = np.zeros((1, self.nHidden))
        self.bVisibleInc = np.zeros((1, self.nFeatures))
        self.bPenInc = np.zeros((1, self.nPen))
        self.bLabelsInc = np.zeros((1, self.nLabels))

        # This code also adds sparcity penalty 
        #self.sparseTarget = 0.2
        #self.sparseTarget2 = 0.1
        #self.sparseCost = 0.001
        #self.sparseDumping = 0.9

        #self.bHidden = 0 * np.log(self.sparseTarget / (1 - self.sparseTarget)) * np.ones((1, self.nHidden))
        #self.hidmeans = self.sparseTarget * np.ones((1, self.nHidden))
        #self.bPen = 0 * np.log(self.sparseTarget2 / (1 - self.sparseTarget2)) * np.ones((1, self.nPen))
        #self.penmeans = self.sparseTarget2 * np.ones((1, self.nPen))
        
        #Last layer
        self.bLabels = bLabels1
        self.wHidPen = wVisHid1;
        self.bPen = bHidden1;
        self.bVisible_l2 = bVisible1;
        self.wLabPen = labelsHidden1;
        
        #first layer
        self.wVisHid = wVisHid0
        self.bVisible = bVisible0
        self.bHidden = bHidden0
        
        self.bHidRec = bHidden0;
        self.bHidden = (bHidden0 + self.bVisible_l2);
        
        self.negativeHiddenProbs = np.random.rand(self.nExamples, self.nHidden);
        self.negativeLabelStates = 1 / self.nLabels * np.ones((self.nExamples, self.nLabels))
        self.data_2 = np.round(np.random.rand(self.nExamples, self.nFeatures))
        self.negativeHiddenProbs = 1 / (1 + np.exp(np.dot(-self.data_2, (2 * self.wVisHid)) - np.tile(self.bHidden, (self.nExamples, 1))))
        
        self.eWeight = self.eWeight / (np.power(1.000015, self.currentEpoch * self.nBatches))
        self.eVisibleBias = self.eVisibleBias / (np.power(1.000015, self.currentEpoch * self.nBatches))
        self.eHiddenBias = self.eHiddenBias / (np.power(1.000015, self.currentEpoch * self.nBatches))

        self.tot = 0;
    
    def continueTD(self, epoch, wVisHid, bVisible, bHidden, wLabPen, bLabels, wHidPen, bPen):
        self.currentEpoch = epoch
        self.wVisHid = wVisHid
        self.bVisible = bVisible
        self.bHidden = bHidden
        self.wLabPen = wLabPen
        self.bLabels = bLabels
        self.wHidPen = wHidPen
        self.bPen = bPen

    def saveState(self, path):
        np.save(path + "/currentEpochD", self.currentEpoch)
        np.save(path + "/wVisHidD", self.wVisHid)
        np.save(path + "/bVisibleD", self.bVisible)
        np.save(path + "/bHiddenD", self.bHidden)
        np.save(path + "/wLabPenD", self.wLabPen)
        np.save(path + "/bLabelsD", self.bLabels)
        np.save(path + "/wHidPenD", self.wHidPen)
        np.save(path + "/bPenD", self.bPen)
        
    def loadState(self, path):
        self.currentEpoch = np.load(path + "/currentEpochD.npy")
        self.wVisHid = np.load(path + "/wVisHidD.npy")
        self.bVisible = np.load(path + "/bVisibleD.npy")
        self.bHidden = np.load(path + "/bHiddenD.npy")
        self.wLabPen = np.load(path + "/wLabPenD.npy")
        self.bLabels = np.load(path + "/bLabelsD.npy")
        self.wHidPen = np.load(path + "/wHidPenD.npy")
        self.bPen = np.load(path + "/bPenD.npy")

    def train(self):
        meanField = mf.MF()

        for epoch in range(self.currentEpoch, self.maxEpoch):
            self.currentEpoch = epoch
            #print "Epoch {}/{}, EpsilonWeight: {}".format(epoch, self.maxEpoch, self.eWeight)

            counter = 0
            rr = np.random.permutation(self.nBatches)
            batch = -1
            errsum = 0
            for batch_rr in rr:
                #Batch_rr is never used, can be used bellow
                batch = batch + 1
                self.tot = self.tot + 1
                self.eWeight = np.max([self.eWeight / 1.000015, 0.0001])
                self.eVisibleBias = np.max([self.eVisibleBias / 1.000015, 0.0001])
                self.eHiddenBias = np.max([self.eHiddenBias / 1.000015, 0.0001])

        ################################ Start of the positive phase ######################################
                data = self.data[batch]
                targets = self.targets[batch]
                data = (data > np.random.rand(self.nExamples, self.nFeatures)).astype(float)

                (self.positiveHiddenProbs, self.positivePenProbs) = meanField.calculate(\
                        data, targets, self.wVisHid, self.bHidden, self.bVisible, self.wHidPen, self.bPen, self.wLabPen, self.bHidRec)
                
                bias_hid = np.tile(self.bHidden, (self.nExamples, 1))
                bias_pen = np.tile(self.bPen, (self.nExamples, 1))
                bias_vis = np.tile(self.bVisible, (self.nExamples, 1))
                bias_lab = np.tile(self.bLabels, (self.nExamples, 1))
                 
                self.positiveProducts = np.dot( np.transpose(data), self.positiveHiddenProbs)
                positiveProductsPen = np.dot( np.transpose(self.positiveHiddenProbs), self.positivePenProbs)
                positiveProductsLabelsPen = np.dot( np.transpose(targets), self.positivePenProbs)
                
                positiveHiddenActive = np.sum(self.positiveHiddenProbs, 0)
                positivePenActive = np.sum(self.positivePenProbs, 0)
                positiveLabelsActive = np.sum(targets, 0)
                positiveVisibleActive = np.sum(data, 0)

        ################################ End of positive phase ############################################
                negativeDataCD1 = 1 / (1 + np.exp(np.dot(-self.positiveHiddenProbs, np.transpose(self.wVisHid)) - bias_vis))
                totin = bias_lab + np.dot(self.positivePenProbs, np.transpose(self.wLabPen))
                positiveLabelProbs1 = np.exp(totin)
                targetOut = positiveLabelProbs1 / (np.dot(np.reshape(np.sum(positiveLabelProbs1, 1), (-1, 1)), np.ones((1,self.nLabels))))
                J = np.argmax(targetOut, 1)
                J1 = np.argmax(targets, 1)
                counter = counter + np.sum(J == J1)

        ################################ Start of the negative phase ######################################
                for iter in range(5):
                    negativeHiddenStates = (self.negativeHiddenProbs > np.random.rand(self.nExamples, self.nHidden)).astype(int)
                    negativePenProbs = 1 / (1 + np.exp( np.dot(-negativeHiddenStates, self.wHidPen) - \
                            np.dot(self.negativeLabelStates, self.wLabPen) - bias_pen))
                    negativePenStates = (negativePenProbs > np.random.rand(self.nExamples, self.nPen)).astype(int)
                    
                    negativeDataProbs = 1 / (1 + np.exp(np.dot(-negativeHiddenStates, np.transpose(self.wVisHid)) - bias_vis))
                    negativeData = (negativeDataProbs > np.random.rand(self.nExamples, self.nFeatures)).astype(int)

                    totin = np.dot(negativePenStates, np.transpose(self.wLabPen)) + bias_lab
                    negativeLabelProbs = np.exp(totin)
                    negativeLabelProbs = negativeLabelProbs / np.dot(np.reshape(np.sum(negativeLabelProbs, 1), (-1,1)), np.ones((1, self.nLabels)))
                  
                    xx = np.cumsum(negativeLabelProbs, 1)
                    xx1 = np.random.rand(self.nExamples,1)
                    self.negativeLabelStates = self.negativeLabelStates * 0
                    for jj in range(self.nExamples):
                        index = np.min(np.where(xx1[jj] <= xx[jj,:]))
                        self.negativeLabelStates[jj, index] = 1
                    xxx = np.sum(self.negativeLabelStates)
  
                    totin = np.dot(negativeData, self.wVisHid) + bias_hid + np.dot(negativePenStates, np.transpose(self.wHidPen))
                    self.negativeHiddenProbs = 1 / (1 + np.exp(-totin))
                
                negativePenProbs = 1 / (1 + np.exp(np.dot(-self.negativeHiddenProbs, self.wHidPen) - \
                        np.dot(negativeLabelProbs, self.wLabPen) - bias_pen))
                
                self.negativeProducts = np.dot(np.transpose(negativeData), self.negativeHiddenProbs)
                negativeProductsPen = np.dot(np.transpose(self.negativeHiddenProbs), negativePenProbs)
                negativeHiddenActive = np.sum(self.negativeHiddenProbs, 0)
                negativePenActive = np.sum(negativePenProbs, 0)
                negativeVisibleActive = np.sum(negativeData, 0)
                negativeLabelActive = np.sum(self.negativeLabelStates, 0)
                negativeProductsLabelPen = np.dot(np.transpose(self.negativeLabelStates), negativePenProbs)

        #################################### End of negative phase ############################################
                err = np.sum( np.square(data - negativeDataCD1) )
                errsum = errsum + err
                if epoch > 5:
                    momentum = self.finalMomentum
                else:
                    momentum = self.initialMomentum
        ##################################### Update weights and biases ########################################
                self.bLabelsInc = momentum * self.bLabelsInc + (self.eVisibleBias / self.nExamples) * \
                        (positiveLabelsActive - negativeLabelActive)

                #self.hidmeans = self.sparseDumping * self.hidmeans + (1 - self.sparseDumping) * positiveHiddenActive / self.nExamples
                #sparseGrads = self.sparseCost * (np.tile(self.hidmeans, (self.nExamples, 1)) - self.sparseTarget)

                #self.penmeans = self.sparseDumping * self.penmeans + (1 - self.sparseDumping) * positivePenActive / self.nExamples
                #sparseGrads2 = self.sparseCost * (np.tile(self.penmeans, (self.nExamples, 1)) - self.sparseTarget2)

                self.labelsPenIncrement = momentum * self.labelsPenIncrement + self.eWeight * (\
                        (positiveProductsLabelsPen - negativeProductsLabelPen) / self.nExamples - \
                        self.weightCost * self.wLabPen)
                
                self.visibleHiddenIncrement = momentum * self.visibleHiddenIncrement + \
                        self.eWeight * ( (self.positiveProducts - self.negativeProducts) / self.nExamples - \
                        self.weightCost * self.wVisHid) #- np.dot(np.transpose(data), sparseGrads) / self.nExamples )
 
                self.bHiddenInc = momentum * self.bHiddenInc + self.eHiddenBias / self.nExamples  * \
                        (positiveHiddenActive - negativeHiddenActive) #- \
                        #self.eHiddenBias / self.nExamples * np.sum(sparseGrads, 0)
                
                self.hiddenPenIncrement = momentum * self.hiddenPenIncrement + self.eWeight * ( \
                        (positiveProductsPen - negativeProductsPen) / self.nExamples - self.weightCost * self.wHidPen) # \
                        #- np.dot(np.transpose(self.positiveHiddenProbs), sparseGrads2) / self.nExamples - \
                        #np.transpose((np.dot(np.transpose(self.positivePenProbs), sparseGrads))) / self.nExamples)
                
                self.bPenInc = momentum * self.bPenInc + self.eHiddenBias / self.nExamples * ( \
                        positivePenActive - negativePenActive) # - self.eHiddenBias / self.nExamples * np.sum(sparseGrads2, 0)
                
                self.bVisibleInc = momentum * self.bVisibleInc + (self.eVisibleBias / self.nExamples) * \
                        (positiveVisibleActive - negativeVisibleActive)
         

                self.wVisHid = self.wVisHid + self.visibleHiddenIncrement
                self.wHidPen = self.wHidPen + self.hiddenPenIncrement
                self.wLabPen = self.wLabPen + self.labelsPenIncrement

                self.bVisible = self.bVisible + self.bVisibleInc
                self.bHidden = self.bHidden + self.bHiddenInc
                self.bPen = self.bPen + self.bPenInc
                self.bLabels = self.bLabels + self.bLabelsInc
            print "End of epoch {}/{}: Reconstruciton error: {}, classError: {}".format(epoch, self.maxEpoch, errsum, 60000-counter)
