import numpy as np

class MF:
    def calculate(self, data, targets, wVisHid, bHidden, bVisible, wHidPen, bPen, wLabPen, bHidRec):
        np.random.seed(444)
        (nFeatures, nHidden) = wVisHid.shape
        (nHidden, nPen) = wHidPen.shape

        nExamples = data.shape[0]
        bias_hid = np.tile(bHidden, (nExamples, 1))
        bias_pen = np.tile(bPen, (nExamples, 1))
        big_bias = np.dot(data, wVisHid)
        lab_bias = np.dot(targets, wLabPen)

        temp_h1 = 1 / (1 + np.exp(np.dot(-data, (2 * wVisHid)) - bias_hid))
        temp_h2 = 1 / (1 + np.exp(np.dot(-temp_h1, wHidPen) - lab_bias - bias_pen))

        for ii in range(10):
            totin_h1 = big_bias + bias_hid + np.dot(temp_h2, np.transpose(wHidPen))
            temp_h1_new = 1 / (1 + np.exp(-totin_h1))

            totin_h2 = np.dot(temp_h1_new, wHidPen) + bias_pen + lab_bias
            temp_h2_new = 1 / (1 + np.exp(-totin_h2))

            diff_h1 = np.sum(np.sum(np.abs(temp_h1_new - temp_h1), 1), 0) / (nExamples * nHidden)
            diff_h2 = np.sum(np.sum(np.abs(temp_h2_new - temp_h2), 1), 0) / (nExamples * nPen)
            
            if diff_h1 < 0.0000001 and diff_h2 < 0.0000001:
                break
            temp_h1 = np.array(temp_h1_new)
            temp_h2 = np.array(temp_h2_new)
        
        temp_h1 = temp_h1_new
        temp_h2 = temp_h2_new

        return (temp_h1, temp_h2)

class ClassificationMF:
    def calculate(self, data, wVisHid, bHidden, bVisible, wHidPen, bPen):
        np.random.seed(777)
        (nFeatures, nHidden) = wVisHid.shape
        (nHidden, nPen) = wHidPen.shape

        nExamples = data.shape[0]
        bias_hid = np.tile(bHidden, (nExamples, 1))
        bias_pen = np.tile(bPen, (nExamples, 1))
        big_bias = np.dot(data, wVisHid)

        temp_h1 = 1 / (1 + np.exp(np.dot(-data, (2 * wVisHid)) - bias_hid))
        temp_h2 = 1 / (1 + np.exp(np.dot(-temp_h1, wHidPen) - bias_pen))

        for ii in range(50):
            totin_h1 = big_bias + bias_hid + np.dot(temp_h2, np.transpose(wHidPen))
            temp_h1_new = 1 / (1 + np.exp(-totin_h1))

            totin_h2 = np.dot(temp_h1_new, wHidPen) + bias_pen
            temp_h2_new = 1 / (1 + np.exp(-totin_h2))

            diff_h1 = np.sum(np.sum(np.abs(temp_h1_new - temp_h1), 1), 0) / (nExamples * nHidden)
            diff_h2 = np.sum(np.sum(np.abs(temp_h2_new - temp_h2), 1), 0) / (nExamples * nPen)
            
            if diff_h1 < 0.0000001 and diff_h2 < 0.0000001:
                break
            temp_h1 = np.array(temp_h1_new)
            temp_h2 = np.array(temp_h2_new)
        
        temp_h1 = temp_h1_new
        temp_h2 = temp_h2_new

        return (temp_h1, temp_h2)


