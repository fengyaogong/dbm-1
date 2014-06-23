import numpy as np

class ClassificationError:
    def __init__(self, nLabels, data, targets, vishid0, visbiases, hidbiases0, vishid1, labhid, labbiases, hidbiases):
        (self.nFeatures, self.nHidden) = vishid0.shape
        self.nLabels = nLabels
        self.data = data
        self.targets = targets
        self.vishid0 = vishid0
        self.visbiases = visbiases
        self.hidbiases0 = hidbiases0
        self.vishid1 = vishid1
        self.labhid = labhid
        self.labbiases = labbiases
        self.hidbiases = hidbiases
        (self.nBatches, self.nExamples, self.nFeatures) = self.data.shape
        self.totExamples = self.nExamples * self.nBatches 
    
    def calculate(self):
        counter = np.zeros((self.nLabels, self.totExamples))
        
        targets_all = np.zeros((self.totExamples, self.nLabels))
        for batch in range(self.nBatches):
            targets_all[batch * self.nExamples:(batch+1) * self.nExamples, :] = self.targets[batch] 

        bias_hid_l0 = np.tile(2 * self.hidbiases0, (self.nExamples, 1))
        bias_pen = np.tile(self.hidbiases, (self.nExamples, 1))

        for batch in range(self.nBatches):
            inter = np.zeros((self.nExamples, self.nLabels))
            data = self.data[batch]

            totinH1 = np.dot(data, (2 * self.vishid0)) + bias_hid_l0
            tempH1 = 1 / (1 + np.exp(-totinH1))

            for tt in range(self.nLabels):
                targets = np.zeros((self.nExamples, self.nLabels))
                targets[:,tt] = 1
                
                lab_bias = np.dot(targets, self.labhid)
                

                temp1 = np.dot(tempH1, np.transpose(self.visbiases)) + np.dot(targets, np.transpose(self.labbiases))
                prod_3 = np.dot(np.ones((self.nExamples, 1)), self.hidbiases) + (np.dot(tempH1, self.vishid1) + np.dot(targets, self.labhid))
                self.prod_3 = prod_3
                self.temp1 = temp1
                p_vl = temp1 + np.reshape(np.sum( np.log(1 + np.exp(prod_3)), 1), (-1,1))
                inter[:, tt] = p_vl[:, 0]

            counter[:, batch*self.nExamples:(batch+1) * 100] = np.transpose(inter)
        
        I = np.max(np.transpose(counter), 1)
        J = np.argmax(np.transpose(counter), 1)
        I1 = np.max(targets_all, 1)
        J1 = np.argmax(targets_all, 1)
        err1 = np.sum(J != J1)
        print "Error: {}".format(err1)

