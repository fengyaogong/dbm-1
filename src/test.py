import numpy as np
import sklearn as sk
import load_mnist as lm
import batches
import rbm
import dbm

np.random.seed(777)

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
maxEpoch = 15
print "\n\nPre training first layer RBM with: {} hidden layers and {} epochs".format(nHidden,maxEpoch)
firstLayerRBM = rbm.FirstLayerRBM(trainBatchData, nHidden, maxEpoch, 0)
firstLayerRBM.train()
print "done"
################################ End of First Layer training #############################
firstLayerRBM.saveState("./savedStates")
################################ Start Last Layer training ###############################
nHiddenPre = nHidden
wVisHidPre = firstLayerRBM.wVisHid
bHiddenPre = firstLayerRBM.bHidden
bVisiblePre = firstLayerRBM.bVisible
nLabels = 10
cd = None
nHidden = 1000
maxEpoch = 15
print "\n\nPre training last layer RBM with: {} hidden layers and {} epochs".format(nHidden,maxEpoch)

#Create batches
(trainBatchData, trainBatchTargets) = batches.getBatches(100, digits, trainImages, trainLabels) 
(testBatchData, testBatchTargets) = batches.getBatches(100, digits, testImages, testLabels)

lastLayerRBM = rbm.LastLayerRBM(nHidden, maxEpoch, nLabels, trainBatchData, trainBatchTargets, nHiddenPre, wVisHidPre, bHiddenPre, bVisiblePre, cd, 0)
lastLayerRBM.train()
################################ End of Last Layer training ###############################
lastLayerRBM.saveState("./savedStates")
################################ Start of dbm training ####################################
nHidden = 500;
nPen = 1000;
maxEpoch = 15;
#Create batches
(trainBatchData, trainBatchTargets) = batches.getBatches(100, digits, trainImages, trainLabels) 
(testBatchData, testBatchTargets) = batches.getBatches(100, digits, testImages, testLabels)
print "\n\nTraining of the DBM"

deepBoltzmannMachine = dbm.DBM(maxEpoch, trainBatchData, trainBatchTargets, nLabels, nHidden, nPen, lastLayerRBM.wVisHid, lastLayerRBM.bLabels, \
        lastLayerRBM.bHidden, lastLayerRBM.bVisible, lastLayerRBM.labelsHidden, firstLayerRBM.wVisHid, firstLayerRBM.bHidden, \ 
        firstLayerRBM.bVisible, 0)
# maxEpoch, data, targets, nLabels, nHidden, nPen, wVisHid1, bLabels1, bHidden1, bVisible1, labelsHidden1, wVisHid0, bHidden0, bVisible0):
deepBoltzmannMachine.train()

deepBoltzmannMachine.saveState("./savedStates")

"""
x = error.ClassificationError(10, trainBatchData, trainBatchTargets, firstLayerRBM.wVisHid, deepBoltzmannMachine.bHidden, firstLayerRBM.bHidden, deepBoltzmannMachine.wHidPen, deepBoltzmannMachine.wLabPen, deepBoltzmannMachine.bLabels, deepBoltzmannMachine.bPen)
585

newC.calc(testBatchData, testBatchTargets, deepBoltzmannMachine.wVisHid, deepBoltzmannMachine.bHidden, deepBoltzmannMachine.bVisible, deepBoltzmannMachine.wHidPen, deepBoltzmannMachine.bPen, deepBoltzmannMachine.wLabPen, deepBoltzmannMachine.bHidRec, deepBoltzmannMachine.bLabels, 100, 600, 10)



"""

