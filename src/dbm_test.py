import numpy as np

class DBM:
    def __init__( maxEpoch, data, targets, nLabels, nHidden, nPen, wVisHid1, bLabels1, bHidden1, bVisible1, labelsHidden1, wVisHid0, bHidden0, bVisible0):
        targets = targets
        maxEpoch = maxEpoch
        eWeight = 0.001   # Learning rate for weights 
        eVisibleBias = 0.001   # Learning rate for biases of visible units 
        eHiddenBias = 0.001   # Learning rate for biases of hidden units 
        weightCost = 0.0002
        initialMomentum = 0.5
        finalMomentum = 0.9
        data = data

        (nBatches, nExamples, nFeatures) = data.shape

        nLabels = nLabels
        nHidden = nHidden
        nPen = nPen        

        currentEpoch = 0
        # Initializing symmetric weights and biases.

        wVisHid = 0.001 * np.random.randn(nFeatures, nHidden);
        wHidPen = 0.001 * np.random.randn(nHidden, nPen);

        wLabPen = 0.001 * np.random.randn(nLabels, nPen);

        bHidden = np.zeros((1, nHidden));
        bVisible  = np.zeros((1, nFeatures));
        bPen = np.zeros((1, nPen));
        bLabels = np.zeros((1, nLabels)); #CHSH
        
        #Probabilities in the positive and negative phase for the hidden layer
        positiveHiddenProbs = np.zeros((nExamples, nHidden))
        negativeHiddenProbs = np.zeros((nExamples, nHidden))
        positivePenProbs = np.zeros((nExamples, nPen))

        #Producst in positive and negative phase
        positiveProducts = np.zeros((nFeatures, nHidden))
        negativeProducts = np.zeros((nFeatures, nHidden))

        
        visibleHiddenIncrement = np.zeros((nFeatures, nHidden))
        hiddenPenIncrement = np.zeros((nHidden, nPen))
        labelsPenIncrement = np.zeros((nLabels, nPen))

        bHiddenInc = np.zeros((1, nHidden))
        bVisibleInc = np.zeros((1, nFeatures))
        bPenInc = np.zeros((1, nPen)) #CHSH
        bLabelsInc = np.zeros((1, nLabels)) #CHSH

        # This code also adds sparcity penalty 
        sparseTarget = 0.2
        sparseTarget2 = 0.1
        sparseCost = 0.001
        sparseDumping = 0.9

        bHidden = np.dot(0 * np.log(sparseTarget / (1 - sparseTarget)), np.ones((1, nHidden)))
        hidmeans = np.dot(sparseTarget, np.ones((1, nHidden)))
        bPen = np.dot(0 * np.log(sparseTarget2 / (1 - sparseTarget2)), np.ones((1, nPen)))
        penmeans = np.dot(sparseTarget2, np.ones((1, nPen)))
        
        #Last layer
        bLabels = bLabels1
        wHidPen = wVisHid;
        bPen = bHidden1;
        bVisible_l2 = bVisible1;
        wLabPen = labelsHidden1;
        
        #first layer
        wVisHid = wVisHid0
        bVisible = bVisible0
        bHidden = bHidden0
        
        bHidRec = bHidden0;
        bHidden = (bHidden0 + bVisible_l2);
        
        negativeHiddenProbs = np.random.rand(nExamples, nHidden); #CHSH 
        negativeLabelStates = 1 / nLabels * np.ones((nExamples, nLabels))
        data_2 = np.round(np.random.rand(nExamples, nFeatures))
        negativeHiddenProbs = 1 / (1 + np.exp(np.dot(-data_2, (2 * wVisHid)) - np.tile(bHidden, (nExamples, 1))))
        
        eWeight = eWeight / (np.power(1.000015, epoch * nBatches))
        eVisibleBias = eVisibleBias / (np.power(1.000015, epoch * nBatches))
        eHiddenBias = eHiddenBias / (np.power(1.000015, epoch * nBatches))

        tot = 0;
        
    def train(:
        for epoch in range(maxEpoch):
            print "Epoch {}/{}".format(epoch, maxEpoch)

            counter = 0
            rr = np.random.permutation(nBatches)
            batch = -1
            errsum = 0
            for batch_rr in rr:
                batch = batch + 1
                tot = tot + 1
                eWeight = np.max([eWeight / 1.000015, 0.0001])
                eVisibleBias = np.max([eVisibleBias / 1.00005, 0.0001])
                eHiddenBias = np.max([eHiddenBias / 1.0005, 0.0001])

        ################################ Start of the positive phase ######################################
                data = data[batch]
                targets = targets[batch]
                data = (data > np.random.rand(nExamples, nFeatures)).astype(int)#CHSH

                (positiveHiddenProbs, positivePenProbs) = #MF something CHSH

                bias_hid = np.tile(bHidden, (nExamples, 1))
                bias_pen = np.tile(bPen, (nExamples, 1))
                bias_vis = np.tile(bVisible, (nExamples, 1))
                bias_lab = np.tile(bLabels, (nExamples, 1))

                positiveProducts = np.dot( np.transpose(data), positiveHiddenProbs)
                positiveProductsPen = np.dot( np.transpose(positiveHiddenProbs), positivePenProbs)
                positiveProductsLabelsPen = np.dot( np.transpose(targets), positivePenProbs)

                positiveHiddenActive = np.sum(positiveHiddenProbs, 0)
                positivePenActive = np.sum(positivePenProbs, 0)
                positiveLabelsActive = np.sum(targets, 0)
                positiveVisibleActive = np.sum(data, 0)

        ################################ End of positive phase ############################################
                negativeDataCD1 = 1 / (1 + np.exp(np.dot(positiveHiddenProbs, np.transpose(wVisHid)) - bias_vis))
                totin = bias_lab + np.dot(positivePenProbs, np.transpose(wLabPen))
                #CHSH up
                positiveLabelProbs1 = np.exp(totin)
                #CHSH Down
                targetOut = positiveLabelProbs1 / (np.dot(np.reshape(np.sum(positiveLabelProbs1, 1), (-1, 1)), np.ones((1,nLabels))))
                J = np.argmax(targetOut, 1)
                J1 = np.argmax(targets, 1)
                #CHSH Down
                counter = counter + np.sum(J == J1)

        ################################ Start of the negative phase ######################################
                for iter in range(5):
                    negativeHiddenStates = (negativeHiddenProbs > np.random.rand(nExamples, nHidden)).astype(int)
                    negativePenProbs = 1 / (1 + np.exp( np.dot(-negativeHiddenStates, wHidPen) - \
                            np.dot(negativeLabelStates, wLabPen) - bias_pen))
                    negativePenStates = (negativePenProbs > np.random.rand(nExamples, nPen)).astype(int)
                    negativeDataProbs = 1 / (1 + np.exp(np.dot(-negativeHiddenStates, np.transpose(wVisHid)) - bias_vis))
                    negativeData = (negativeDataProbs > np.random.rand(nExamples, nFeatures)).astype(int)

                    totin = np.dot(negativePenStates, np.transpose(wLabPen)) + bias_lab
                    negativeLabelProbs = np.exp(totin)
                    #CHSH Down
                    negativeLabelProbs = negativeLabelProbs / np.dot(np.reshape(np.sum(negativeLabelProbs, 1), (-1, 1)), np.ones((1, nLabels)))
                  
                    xx = np.cumsum(negativeLabelProbs, 1)
                    xx1 = np.random.rand(nExamples,1)
                    negativeLabelStates = negativeLabelStates * 0
                    for jj in range(nExamples):
                        #TODO: check this
                        index = np.min(np.where(xx1[jj] <= xx[jj,:]))
                        negativeLabelStates[jj, index] = 1
                    xxx = np.sum(negativeLabelStates)
  
                    totin = np.dot(negativeData, wVisHid) + bias_hid + np.dot(negativePenStates, np.transpose(wHidPen))
                    negativeHiddenProbs = 1 / (1 + np.exp(-totin))
                
                negativePenProbs = 1 / (1 + np.exp(np.dot(-negativeHiddenProbs, wHidPen) - \
                        np.dot(negativeLabelProbs, wLabPen) - bias_pen))

                negativeProducts = np.dot(np.transpose(negativeData), negativeHiddenProbs)
                negativeProductsPen = np.dot(np.transpose(negativeHiddenProbs), negativePenProbs)
                negativeHiddenActive = np.sum(negativeHiddenProbs, 0)
                negativePenActive = np.sum(negativePenProbs, 0)
                negativeVisibleActive = np.sum(negativeData, 0)
                negativeLabelActive = np.sum(negativeLabelStates, 0)
                negativeProductsLabelPen = np.dot(np.transpose(negativeLabelStates), negativePenProbs)

        #################################### End of negative phase ############################################
                err = np.sum( np.square(data - negativeDataCD1) )
                errsum = errsum + err

                if epoch > 5:
                    momentum = finalMomentum
                else:
                    momentum = initialMomentum
        ##################################### Update weights and biases ########################################
               #CHSH doesn't exist
               
#CHSH lbals
                bLabelsInc = momentum * bLabelsInc + (eVisibleBias / nExamples) * \
                        (positiveLabelsActive - negativeLabelActive)

                hidmeans = np.dot(sparseDumping, hidmeans) + np.dot(1 - sparseDumping, positiveHiddenActive) / nExamples
                sparseGrads = np.dot(sparseCost, np.tile(hidmeans, (nExamples, 1)) - sparseTarget)

                penmeans = np.dot(sparseDumping, penmeans) + np.dot(1 - sparseDumping, positivePenActive) / nExamples
                sparseGrads2 = np.dot(sparseCost, np.tile(penmeans, (nExamples, 1)) - sparseTarget2)
                
#CHSH brace
                labelsPenIncrement = momentum * labelsPenIncrement + eWeight * ( \
                        (positiveProductsLabelsPen - negativeProductsLabelPen) / nExamples - \
                        weightCost * wLabPen)
                
                wVisHid = wVisHid + visibleHiddenIncrement
                
                visibleHiddenIncrement = momentum * visibleHiddenIncrement + \
                        eWeights * ( (positiveProducts - negativeProducts) / nExamples - \
                        weightCost * wVisHid - np.dot(np.transpose(data), sparseGrads) / nExamples )
                
                #CHSH biases
                bHiddenInc = momentum * bHiddenInc + (eHiddenBias / nExamples) * \
                        (positiveHiddenActive - negativeHiddenActive) - \
                        eHiddenBias / nExamples * np.sum(sparseGrads)
                
                #CHSH Down - currently doesn't exist
                bVisibleInc = momentum * bVisibleInc + (eVisibleBias / nExamples) * \
                        (positiveVisibleActive - negativeVisibleActive) - 


      
                bVisible = bVisible + bVisibleInc
                bHidden = bHidden + bHiddenInc
                bLabels = bLabels + bLbalesInc

