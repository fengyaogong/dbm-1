import numpy as np

class ClassificationError:
    def __init__( nLabels, data, targets, vishid0, visbiases, hidbiases0, vishid1, labhid, labbiases, hidbiases):
        (nFeatures, nHidden) = vishid0.shape
        nLabels = nLabels
        data = data
        targets = targets
        vishid0 = vishid0
        visbiases = visbiases
        hidbiases0 = hidbiases0
        vishid1 = vishid1
        labhid = labhid
        labbiases = labbiases
        hidbiases = hidbiases
        (nBatches, nExamples, nFeatures) = data.shape
        totExamples = nExamples * nBatches 
    
    def calculate(:
        counter = np.zeros((nLabels, totExamples))
        
        targets_all = np.zeros((totExamples, nLabels))
        for batch in range(nBatches):
            targets_all[batch * nExamples:(batch+1) * nExamples, :] = targets[batch] 

        bias_hid_l0 = np.tile(2 * hidbiases0, (nExamples, 1))
        bias_pen = np.tile(hidbiases, (nExamples, 1))

        for batch in range(nBatches):
            inter = np.zeros((nExamples, nLabels))
            data = data[batch]

            totinH1 = np.dot(data, (2 * vishid0)) + bias_hid_l0
            tempH1 = 1 / (1 + np.exp(-totinH1))

            for tt in range(nLabels):
                targets = np.zeros((nExamples, nLabels))
                targets[:,tt] = 1
                
                lab_bias = np.dot(targets, labhid)

                temp1 = np.dot(tempH1, np.transpose(visbiases)) + np.dot(targets, np.transpose(labbiases))
                prod_3 = np.dot(np.ones((nExamples, 1)), hidbiases) + (np.dot(tempH1, vishid1) + np.dot(targets, labhid))
                p_vl = temp1 + np.sum( np.log(1 + np.exp(prod_3)), 1)
                inter[:, tt] = p_vl

            counter[:, batch*nExamples:(batch+1) * 100] = np.transpose(inter)
        
        I = np.max(np.transpose(counter), 1)
        J = np.argmax(np.transpose(counter), 1)
        I1 = np.max(targets_all, 1)
        J1 = np.argmax(targets_all, 1)
        err1 = np.sum(J != J1)
        print "Error: {}".format(err1)

