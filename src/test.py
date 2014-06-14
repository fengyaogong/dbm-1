import numpy as np
import sklearn as sk
import load_mnist as lm
import batches
import rbm

np.random.seed(100)

#Digits to be worked with
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Convert input to python format
(testImages, testLabels) = lm.read(digits, dataset = "testing", path = "../data/")
(trainImages, trainLabels) = lm.read(digits, dataset = "training", path = "../data/")

#Create batches
(trainBatchData, trainBatchTargets) = batches.getBatches(100, digits, trainImages, trainLabels) 
(testBatchData, testBatchTargets) = batches.getBatches(100, digits, testImages, testLabels)

#Get number of: examples, features and batches for train data
(nBatches, nExamples, nFeatures) = trainBatchData.shape 

################################ Train First Layer #######################################
#Number of hiden layers and max number of epoch for training
nHidden = 500
maxEpoch = 5
print "Pre training first layer RBM with: {} hidden layers and {} epochs".format(nHidden,maxEpoch)
firstLayerRBM = rbm.FirstLayerRBM(trainBatchData, nHidden, maxEpoch)
firstLayerRBM.train()
print "done"

