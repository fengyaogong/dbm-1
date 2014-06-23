import numpy as np
import mf

class BackpropOptimization:
    def __init__(self, maxEpoch, nLabels, data, testData, targets, testTargets, wVisHidD, wHidPenD, bHiddenD, bVisibleD, bPenD):
        mfClass = mf.ClassificationMF()
        self.data = data
        self.testData = testData
        self.targets = targets
        self.testTargets = testTargets
        self.maxEpoch = maxEpoch
        self.testErr = np.zeros((maxEpoch, 1))
        self.testCErr = np.zeros((maxEpoch, 1))
        self.trainErr = np.zeros((maxEpoch, 1))
        self.trainCErr = np.zeros((maxEpoch, 1))

        (self.nBatches, self.nExamples, self.nFeatures) = self.data.shape
        self.N = self.nExamples

        (self.nFeatures, self.nHidden) = wVisHidD.shape
        (self.nHidden, self.nPen) = wHidPenD.shape

        (self.nBatchesTest, self.nExamplesTest, self.nFeaturesTest) = self.testData.shape
        self.N = self.nExamplesTest
        
        self.wVisHid = wVisHidD
        self.wHidPen = wHidPenD
        self.bHidden = bHiddenD
        self.bVisible = bVisibleD
        self.bPen = bPenD
        self.nLabels = nLabels

        self.temp_h2_test = np.zeros((self.nExamplesTest, self.nPen, self.nBatchesTest))
        for batch in range(self.nBatchesTest):
            dataN = self.testData[batch]
            (temp_h1, temp_h2) = mfClass.calculate(dataN, self.wVisHid, self.bHidden, self.bVisible, self.wHidPen, self.bPen)
            self.temp_h2_test[:, :, batch] = temp_h2

        self.temp_h2_train = np.zeros((self.nExamples, self.nPen, self.nBatches))
        for batch in range(self.nBatches):
            dataN = self.data[batch]
            (temp_h1, temp_h2) = mfClass.calculate(dataN, self.wVisHid, self.bHidden, self.bVisible, self.wHidPen, self.bPen)
            self.temp_h2_train[:, :, batch] = temp_h2



        self.w1_penhid = np.transpose(self.wHidPen)
        self.w1_vishid = self.wVisHid
        self.w2 = self.wHidPen
        self.h1_biases = self.bHidden
        self.h2_biases = self.bPen
        
        self.w_class = 0.1 * np.random.randn(self.nPen, self.nLabels)
        self.topbiases = 0.1 * np.random.randn(1, self.nLabels)

    def calculate(self):
        mfClass = mf.ClassificationMF()
        for epoch in range(self.maxEpoch):
            ##################Test stats######################
            N = self.nExamplesTest
            bias_hid = np.tile(self.h1_biases, (N, 1))
            bias_pen = np.tile(self.h2_biases, (N, 1))
            bias_top = np.tile(self.topbiases, (N, 1))
            
            err_cr = 0
            counter = 0
            
            for batch in range(self.nBatchesTest):
                data = self.testData[batch]
                temp_h2 = self.temp_h2_test[:, :, batch]
                target = self.testTargets[batch]
                
                w1probs = 1 / (1 + np.exp(np.dot(-data, self.w1_vishid) - np.dot(temp_h2, self.w1_penhid) - bias_hid))
                w2probs = 1 / (1 + np.exp(np.dot(-w1probs, self.w2) - bias_pen))
                targetout = np.exp(np.dot(w2probs, self.w_class) + bias_top)
                targetout = targetout / np.tile(np.reshape(np.sum(targetout, 1), (-1,1)), (1, self.nLabels))
                J = np.argmax(targetout, 1)
                J1 = np.argmax(target, 1)
                counter = counter + np.sum(J != J1)
                err_cr = err_cr - np.sum(target * np.log(targetout))
            
            self.testErr[epoch] = counter
            self.testCErr[epoch] = err_cr

            print "Test => Epoch: {}, classError: {}, CEError: {}".format(epoch, counter, err_cr)

            ###################Train stats#####################
            N = self.nExamples
            bias_hid = np.tile(self.h1_biases, (N, 1))
            bias_pen = np.tile(self.h2_biases, (N, 1))
            bias_top = np.tile(self.topbiases, (N, 1))
            
            err_cr = 0
            counter = 0
            
            for batch in range(self.nBatches):
                data = self.data[batch]
                temp_h2 = self.temp_h2_train[:, :, batch]
                target = self.targets[batch]
                
                w1probs = 1 / (1 + np.exp(np.dot(-data, self.w1_vishid) - np.dot(temp_h2, self.w1_penhid) - bias_hid))
                w2probs = 1 / (1 + np.exp(np.dot(-w1probs, self.w2) - bias_pen))
                targetout = np.exp(np.dot(w2probs, self.w_class) + bias_top)
                targetout = targetout / np.tile(np.reshape(np.sum(targetout, 1), (-1,1)), (1, self.nLabels))
                J = np.argmax(targetout, 1)
                J1 = np.argmax(target, 1)
                counter = counter + np.sum(J != J1)
                err_cr = err_cr - np.sum(target * np.log(targetout))
            
            self.trainErr[epoch] = counter
            self.trainCErr[epoch] = err_cr

            print "Train => Epoch: {}, classError: {}, CEError: {}".format(epoch, counter, err_cr)
"""
            #####################Conjugate Gradient Optimization#################
            rr = np.random.permutation(self.nBatches)
            for batch in range(self.nBatches / 100):
                data = np.zeros((10000, self.nFeatures))
                temp_h2 = np.zeros((10000, self.nPen))
                targets = np.zeros((10000, self.nLabels))
                tt1 = range(batch * 100, (batch + 1) * 100) 
                for tt in range(100):
                    data[tt * 100:(tt+1) * 100, :] = self.data[rr[tt1[tt]]]
                    temp_h2[tt * 100:(tt+1) * 100, :] = self.temp_h2_train[:, :, rr[tt1[tt]]]
                    targets[tt * 100:(tt+1) * 100, :] = self.targets[rr[tt1[tt]]]

                #############CG with 3 line searches#############
                VV = np.append(self.w1_vishid.reshape(-1), self.w1_penhid.reshape(-1), self.w2.reshape.reshape(-1), self.w_class.reshape(-1), \
                        self.h1_biases.reshape(-1), self.h2_biases.reshape(-1), self.topbiases.reshape(-1))
                Dim = np.append([self.nFeatures], [self.nHidden], [self.nPen], axis=0)

                max_iter = 3
                if epoch < 6:


        
"""
