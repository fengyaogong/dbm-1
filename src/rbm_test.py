import numpy as np

class FirstLayerRBM:
    def __init__( data, nHidden, maxEpoch):
        nHidden = nHidden
        maxEpoch = maxEpoch
        CD = 1
        data = data

        #e - epsilon - leraning rate
        eWeights = 0.05
        eVisibleBiases = 0.05
        eHiddenBiases = 0.05
        
        #TODO: what are this
        weightCost = 0.001
        initialMomentum = 0.5
        finalMomentum = 0.9
        
        #Number of: examples (in one batch), features and batches in the current data set
        (nBatches, nExamples, nFeatures) = data.shape
        
        #In which epoch of the training algorithm are is the program
        currentEpoch = 0

        #Visible -> Hidden weights
        wVisHid = 0.001 * np.random.randn(nFeatures, nHidden)
        
        #Biases for hidden and visible layer
        bHidden = np.zeros((1, nHidden))
        bVisible = np.zeros((1, nFeatures))
        
        #Probabilities in the positive and negative phase for the hidden layer
        positiveHiddenProbs = np.zeros((nExamples, nHidden))
        negativeHiddenProbs = np.zeros((nExamples, nHidden))
        
        #Producst in positive and negative phase
        positiveProducts = np.zeros((nFeatures, nHidden))
        negativeProducts = np.zeros((nFeatures, nHidden))
        
        #TODO:
        visibleHiddenIncrement = np.zeros((nFeatures, nHidden))

        #TODO: Check if this are increments for visible and hidden biases
        bHiddenInc = np.zeros((1, nHidden))
        bVisibleInc = np.zeros((1, nFeatures))
        
        #Positive probabilities for every batch,example and hidden unit
        batchPositiveHiddenProbs = np.zeros((nBatches, nExamples, nHidden))
        
    def train(:
        for epoch in range(currentEpoch, maxEpoch):
            currentEpoch = epoch
            print "Current epoch: {}, max epoch: {}".format(currentEpoch, maxEpoch)

            #TODO: what is this
            errsum = 0

            for batch in range(nBatches):
                #Bias for every example in one batch
                bVisibleOne = np.tile(bVisible, (nExamples, 1))
                bHiddenOne = np.tile(2 * bHidden, (nExamples, 1))
                
         ########################## Start of the positive phase ################################       
                #Convert the input data from one batch to 1/0 
                data = data[batch]
                data = (data > np.random.rand(nExamples, nFeatures)).astype(int)

                positiveHiddenProbs = 1 / (1 + np.exp(np.dot(-data, (2 * wVisHid)) - bHiddenOne))
                batchPositiveHiddenProbs[batch] = positiveHiddenProbs
                
                positiveProducts = np.dot(np.transpose(data), positiveHiddenProbs)
                positiveHiddenActive = np.sum(positiveHiddenProbs, 0)
                positiveVisibleActive = np.sum(data, 0)
        ########################## End of positive phase ########################################

        ########################## Start of the negative phase ##################################
                positiveHiddenStates = (positiveHiddenProbs > np.random.rand(nExamples, nHidden)).astype(int)
                #Reconstruction of data
                negativeData = 1 / (1 + np.exp(np.dot(-positiveHiddenStates, np.transpose(wVisHid)) - bVisibleOne))
                negativeData = (negativeData > np.random.rand(nExamples, nFeatures)).astype(int)
                negativeHiddenProbs = 1 / (1 + np.exp(np.dot(-negativeData, (2 * wVisHid)) - bHiddenOne))

                negativeProducts = np.dot(np.transpose(negativeData), negativeHiddenProbs)
                negativeHiddenActive = np.sum(negativeHiddenProbs, 0)
                negativeVisibleActive = np.sum(negativeData, 0)

        ########################## End of negative phase #######################################
                err = np.sum( np.square(data - negativeData) )
                errsum = errsum + err

                if epoch > 5:
                    momentum = finalMomentum
                else:
                    momentum = initialMomentum

        ########################## Update weights and biases ###################################
                visibleHiddenIncrement = momentum * visibleHiddenIncrement + \
                        eWeights * ( (positiveProducts - negativeProducts) / nExamples - \
                        weightCost * wVisHid)
                bVisibleInc = momentum * bVisibleInc + (eVisibleBiases / nExamples) * \
                        (positiveVisibleActive - negativeVisibleActive)
                bHiddenInc = momentum * bHiddenInc + (eHiddenBiases / nExamples) * \
                        (positiveHiddenActive - negativeHiddenActive)
                
                wVisHid = wVisHid + visibleHiddenIncrement
                bVisible = bVisible + bVisibleInc
                bHidden = bHidden + bHiddenInc
        ########################## End of weights and biases updates ###########################

            print "End of epoch: {}, errsum: {}".format(epoch, errsum)


class LastLayerRBM:
    def __init__( nHidden, maxEpoch, data, targets, nHiddenPre, wVisHidPre, bHiddenPre, bVisiblePre, cd):
        nHidden = nHidden
        maxEpoch = maxEpoch
        if cd == None:
            CD = -1
        else:
            CD = cd
        #The number of steps for CD, in fact this is calculated as 
        # NumberOfSteps = current_epoch / nStepsCD
        nStepsCD = 20
        data = data
        targest = targets

        #e - epsilon - leraning rate
        eWeights = 0.05
        eVisibleBiases = 0.05
        eHiddenBiases = 0.05
        
        #TODO: what are this
        weightCost = 0.001
        initialMomentum = 0.5
        finalMomentum = 0.9
        
        #Previous layer weights and biases
        wVisHidPre = wVisHidPre
        bHiddenPre = bHiddenPre
        bVisiblePre = bVisiblePre

        #Number of: examples (in one batch), features and batches in the current data set
        (nBatches, nExamples, nFeatures) = data.shape
       
        #The hidden layer from previous 'layer' becomes the visible layer of this
        nFeaturesPre = nFeatures
        nFeatures = nHiddenPre

        #In which epoch of the training algorithm are is the program
        currentEpoch = 0

        #Visible -> Hidden weights
        wVisHid = 0.01 * np.random.randn(nFeatures, nHidden)
        
        #Biases for hidden and visible layer
        bHidden = np.zeros((1, nHidden))
        bVisible = np.zeros((1, nFeatures))
        
        #Probabilities in the positive and negative phase for the hidden layer
        positiveHiddenProbs = np.zeros((nExamples, nHidden))
        negativeHiddenProbs = np.zeros((nExamples, nHidden))
        
        #Producst in positive and negative phase
        positiveProducts = np.zeros((nFeatures, nHidden))
        negativeProducts = np.zeros((nFeatures, nHidden))
        
        #TODO:
        visibleHiddenIncrement = np.zeros((nFeatures, nHidden))

        #TODO: Check if this are increments for visible and hidden biases
        bHiddenInc = np.zeros((1, nHidden))
        bVisibleInc = np.zeros((1, nFeatures))
        
        nLabels = nLabels
        labelsHidden = 0.01 * np.random.randn(nLabels, nHidden)
        bLabels = np.zeros((1, nLabels))
        labelsHiddenInc = np.zeros((nLabels, nHidden))
        bLbalesInc = np.zeros((1, nLabels))
        

    def train(:
        for epoch in range(maxEpoch):
            print "Current epoch {}/{}".format(epoch, maxEpoch)

            if CD == None:
                CD = np.ceil((epoch + 1) / float(nStepsCD))
        
            eWeights = eWeights / float(CD)
            eVisibleBiases = eVisibleBiases / float(CD)  
            eHiddenBiases = eHiddenBiases / float(CD)

            errsum = 0
            for batch in range(nBatches):
        ############################ Start of the positive phase ###############################
                dataPre = data[batch]
                positiveHiddenProbsPre = 1 / (1 + np.exp(np.dot(-dataPre, 2 * wVisHidPre) - np.tile(2 * bHiddenPre, (nExamples, 1))))
                data = positiveHiddenProbsPre > np.random.rand(nExamples, nFeatures)
                #TODO: check are these the real targets
                targets = targets[batch] #chsh
                
                bHiddenOne = np.tile(bHidden, (nExamples, 1))
                bVisibleOne = np.tile(bVisible, (nExamples, 1))
                bLabelsOne = np.tile(bLbales, (nExamples, 1))

                #TODO: change name of labHid to wLabelsHidden
                positiveHiddenProbs = 1 / (1 + np.exp(np.dot(-data, wVisHid) - np.dot(targets, labelsHidden) - bHiddenOne))
                positiveProducts = np.dot( np.transpose(data), positiveHiddenProbs )
                positiveProductsLabelsHidden = np.dot( np.transpose(targets), positiveHiddenProbs )

                positiveHiddenActive = np.sum(positiveHiddenProbs, 0)
                positiveVisibleActive = np.sum(data, 0)
                positiveLabelsActive = np.sum(targets, 0)
        ############################ End of positive phase #####################################
                positiveHiddenProbs_tmp = positiveHiddenProbs
        ############################ Start of the negative phase ###############################
                for iterCD in range(CD):
                    positiveHiddenStates = positiveHiddenProbs_tmp > np.random.rand(nExamples, nHidden)
                    totin = np.dot(positiveHiddenStates, np.transpose(labelsHidden)) + bLabels
                    negativeLabelProbs = np.exp(totin)
#chsh
                    negativeLabelProbs = negativeLabelProbs / ( np.dot(np.reshape(np.sum(negativeLabelProbs, 1), (-1, 1)), np.ones((1, nLabels))))

                    xx = np.cumsum(negativeLabelProbs, 1)
                    xx1 = np.random.rand(nExamples,1) #chsh
                    negativeLabelStates = negativeLabelProbs * 0
                    for jj in range(nExamples):
                        #TODO: check this
                        index = np.min(np.where(xx1[jj] <= xx[jj,:]))
                        negativeLabelStates[jj, index] = 1
                    xxx = np.sum(negativeLabelStates) #chsh

                    negativeData = 1 / (1 + np.exp(np.dot(-positiveHiddenStates, np.transpose(2 * wVisHid)) - bVisibleOne))
                    #chsh
                    negativeData = negativeData > np.random.rand(nExamples, nFeatures)

                    positiveHiddenProbs_tmp = 1 / (1 + np.exp(np.dot(-negativeData, wVisHid) - np.dot(negativeLabelStates, labelsHidden) - 
                        bHiddenOne))

                negativeHiddenProbs = positiveHiddenProbs_tmp

                negativeProducts = np.dot(np.transpose(negativeData), negativeHiddenProbs)
                negativeHiddenActive = np.sum(negativeHiddenProbs, 0)
                negativeVisibleActive = np.sum(negativeData, 0)
                negativeLabelActive = np.sum(negativeLabelStates, 0)
                negativeProductsLabelHidden = np.dot(np.transpose(negativeLabelStates), negativeHiddenProbs)
        ############################ End of negative phase #######################################
                err = np.sum( np.square(data - negativeData) )
                errsum = errsum + err

                if epoch > 5:
                    momentum = finalMomentum
                else:
                    momentum = initialMomentum
 
        ############################ Weight updates ##############################################
                visibleHiddenIncrement = momentum * visibleHiddenIncrement + \
                        eWeights * ( (positiveProducts - negativeProducts) / nExamples - \
                        weightCost * wVisHid)
                labelsHiddenInc = momentum * labelsHiddenInc + \
                        eWeights * ( (positiveProductsLabelsHidden - negativeProductsLabelHidden) / nExamples \
                        - weightCost * labelsHidden)

                bVisibleInc = momentum * bVisibleInc + (eVisibleBiases / nExamples) * \
                        (positiveVisibleActive - negativeVisibleActive)
                bHiddenInc = momentum * bHiddenInc + (eHiddenBiases / nExamples) * \
                        (positiveHiddenActive - negativeHiddenActive)
                bLbalesInc = momentum * bLbalesInc + (eVisibleBiases / nExamples) * \
                        (positiveLabelsActive - negativeLabelActive)

                wVisHid = wVisHid + visibleHiddenIncrement
                #chsh
                labelsHidden = labelsHidden + labelsHiddenInc 
                
                bVisible = bVisible + bVisibleInc
                bHidden = bHidden + bHiddenInc
                bLabels = bLabels + bLbalesInc

            print "End of epoch: {}, errsum: {}".format(epoch, errsum)
