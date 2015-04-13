import random
import math

# data and hypothesis globals
ELEMENT_COUNT = 500
WEIGHT_COUNT = 11
trainingSet = []
testSet = []
theta = []

# Data and hypothesis Initialization
def init():
    global trainingSet
    global testSet
    trainingSet = []
    testSet = []
    for n in range(ELEMENT_COUNT):
        trainingSet.append([])
        testSet.append([])
        for m in range(WEIGHT_COUNT):
            trainingSet[n].append(1 if random.randint(0,1) == 1 else -1)
            testSet[n].append(1 if random.randint(0,1) == 1 else -1)

# hypothesis function for logistic regression
def h(theta, dataVector):
    dot = 0
    for i in range(WEIGHT_COUNT):
        dot += theta[i] * dataVector[i]
    return 1/(1 + math.exp(-dot))

# Stochastic Gradient Descent
def sgd(data, labels, theta, alpha = 0.1, track = False, epochs = 1):
    weights = []
    y = 1
    for e in range(epochs):
        for i in range(ELEMENT_COUNT):
            weights.append(list(theta))
            for j in range(WEIGHT_COUNT):
                theta[j] = theta[j] + alpha * (labels[i] - h(theta, data[i])) * data[i][j]
    if track:
        return weights

# first component labels
def partA():
    global trainingSet
    global testSet
    global theta

    trainingLabels = []
    testLabels = []
    theta = [1 for i in range(WEIGHT_COUNT)]

    # initialize labels
    for n in range(ELEMENT_COUNT):
        trainingLabels.append(1 if trainingSet[n][0] == 1 else 0)
        testLabels.append(1 if testSet[n][0] == 1 else 0)

    print ('start', theta)
    sgd(trainingSet, trainingLabels, theta)
    print ('end', theta)

# half feature labels
def partB():
    global trainingSet
    global testSet
    global theta

    trainingLabels = []
    testLabels = []
    theta = [1 for i in range(WEIGHT_COUNT)]

    # initialize labels
    for n in range(ELEMENT_COUNT):
        t1 = sum(trainingSet[n])
        t2 = sum(testSet[n])
        trainingLabels.append(1 if t1 >= 0 else 0)
        testLabels.append(1 if t2 >= 0 else 0)

    print ('start', theta)
    sgd(trainingSet, trainingLabels, theta)
    print ('end', theta)

# used for partC to initialize labels
def noiseDistribution(dataVector):
    return random.uniform(-4, 4) + sum(dataVector)

# calculats the log-likelihood of the given weight vector theta
def logLikelihood(data, labels, theta):
    sum = 0
    for i in range(len(data)):
        htheta = h(theta, data[i])
        # TODO: getting negative number for log
        sum += labels[i] * math.log(htheta) + (1 - labels[i]) * math.log(1 - htheta)
    return sum

# noisy labels
def partC():
    global trainingSet
    global testSet
    global theta

    trainingLabels = []
    testLabels = []
    theta = [1 for i in range(WEIGHT_COUNT)]

    # initialize labels
    for n in range(ELEMENT_COUNT):
        t1 = sum(trainingSet[n])
        t2 = sum(testSet[n])
        trainingLabels.append(1 if noiseDistribution(trainingSet[n]) > 0 else 0)
        testLabels.append(1 if noiseDistribution(testSet[n]) > 0 else 0)

    print ('start', theta)
    weights = sgd(trainingSet, trainingLabels, theta, 0.1, True, 2)
    print ('end', theta)
    print('total weights', len(weights))

    # calculate average of weights
    wAvg = [0 for i in range(WEIGHT_COUNT)]
    for j in range(WEIGHT_COUNT):
        for i in range(len(weights)):
            wAvg[j] += weights[i][j]
        wAvg[j] = wAvg[j]/len(weights)
    # calculate average of second epoch
    wAvg2 = [0 for i in range(WEIGHT_COUNT)]
    for j in range(WEIGHT_COUNT):
        for i in range((len(weights)/2)-1, len(weights)):
            wAvg2[j] += weights[i][j]
        wAvg2[j] = wAvg[j]/(len(weights)/2)

    print('final weight vector', weights[len(weights)-1])
    print('total average', wAvg)
    print('second epoch average', wAvg2)
    print('likelihood fwv', logLikelihood(trainingSet, trainingLabels, weights[len(weights)-1]))
    print('likelihood ta', logLikelihood(trainingSet, trainingLabels, wAvg))
    print('likelihood ea', logLikelihood(trainingSet, trainingLabels, wAvg2))


###############
# PROGRAM START
###############
init()
#partA()
#partB()
partC()