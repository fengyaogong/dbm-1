import numpy as np

def getBatches(batchSize, digits, data, labels):
    targets = []
    target = np.zeros(len(digits))
    
    for one in labels:
        target[one] = 1
        targets.append(np.array(target))
        #Return back the target to 0
        target[one] = 0
    
    targets = np.array(targets)
    data = data / 255.0
    dataLen = len(data)

    print "The length of the data set is: {}".format(dataLen)
    np.random.seed(0)
    randomOrder = np.random.permutation(dataLen)
    numBatches = dataLen / batchSize
    numFeatures = len(data[0][0]) * len(data[0])

    batchData = np.zeros([numBatches, batchSize, numFeatures])
    batchTargets = np.zeros([numBatches, batchSize, len(digits)])
    
    print randomOrder[0]

    for one in range(numBatches):
        batchData[one, :, :] = np.reshape(data[randomOrder[0 + batchSize * one:batchSize + batchSize * one]], (batchSize, numFeatures))
        batchTargets[one, :, :] = targets[randomOrder[0 + batchSize * one:batchSize + batchSize * one]]
   
    return batchData, batchTargets.astype(int)
