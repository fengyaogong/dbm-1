import numpy as np
import mf
def calc(inData, inTargets, wVisHid, bHidden, bVisible, wHidPen, bPen, wLabPen, bHidRec, bLabels, nExamples, nBatches, nLabels):
    meanField = mf.MF()
    counter = 0
    for batch in range(nBatches):
        data = inData[batch]
        targets = inTargets[batch]
        (positiveHiddenProbs, positivePenProbs) = meanField.calculate(\
                data, targets, wVisHid, bHidden, bVisible, wHidPen, bPen, wLabPen, bHidRec)
    
        bias_lab = np.tile(bLabels, (nExamples, 1))

        totin = bias_lab + np.dot(positivePenProbs, np.transpose(wLabPen))
        positiveLabelProbs1 = np.exp(totin)
        targetOut = positiveLabelProbs1 / (np.dot(np.reshape(np.sum(positiveLabelProbs1, 1), (-1, 1)), np.ones((1,nLabels))))
        J = np.argmax(targetOut, 1)
        J1 = np.argmax(targets, 1)
        counter = counter + np.sum(J == J1)

    print counter
