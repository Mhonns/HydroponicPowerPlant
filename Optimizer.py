"""Python code for optimizing the lettuce growth rate using Neural network
the neural structure will look like:

[]  - each node
0   - output layer
w   - weight
hn  - hidden layer at n

h1         h2                       hn      o layer
[[w, w, w] [[[w, w, w, w, w, w, w]   ...    [[w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]   [w, w, w, w, w, w, w]   ...     [w, w, w, w, w, w, w]
 [w, w, w]]  [w, w, w, w, w, w, w]]  ...]    [w, w, w, w, w, w, w]
                                             [w, w, w, w, w, w, w]
                                             [w, w, w, w, w, w, w]
                                             [w, w, w, w, w, w, w]]

                   Created by Nathadon Samairat and PMKFB team 12 Mar 2023
"""

import numpy as np

intputSize = 3          # the number of input nodes  (light, ph, ec)
outputSize = 10         # the number of output nodes (px per day)
hiddenSize = 7          # the number of nodes in hiddenlayers
hiddenLayerSize = 3     # the number of the hidden layers
haveSaved = False       # true to use the weight and bias from weightData.txt
save = True            # admit to save the weight and bias or not

# assign all weights and bias
def initializeNetwork():
    weightTensor = np.array(
        [[np.random.uniform(-1, 1, intputSize) for _ in range(hiddenSize)]] +
        [[[np.random.uniform(-1, 1, hiddenSize) for _ in range(hiddenSize)] for _ in range(hiddenLayerSize - 1)]] +
        [[np.random.uniform(-1, 1, hiddenSize) for _ in range(outputSize)]], dtype=object
    )
    biasTensor = np.array(
        [[np.zeros_like(1) for _ in range(hiddenSize)]] +
        [[[np.zeros_like(1) for _ in range(hiddenSize)] for _ in range(hiddenLayerSize - 1)]] +
        [[np.zeros_like(1) for _ in range(outputSize)]], dtype=object
    )
    return weightTensor, biasTensor

# activation function sigmoid and Relu in this case
# Args
#    x = numpy array of inactive nodes
def sigmoidIng(x):
    return 1 / (1 + np.exp(-x))
def ReLUIng(x):
    return np.maximum(0, x)
def softmax(x):
    return (np.exp(x)/np.exp(x).sum())

# Forward propagation
# Calculate node activation using 
# activation = sum(weight * input) + bias
# Args
#       weightTensor = current weight tensor
#       inputData    = input data from input layer
def activateNodes(weightTensor, bias, inputData):
    hiddenActive = np.zeros((hiddenLayerSize, hiddenSize), dtype=float)
    outputActive = np.zeros((outputSize), dtype=float)
    activeNode = ReLUIng(np.matmul(weightTensor[0], inputData) + bias[0])
    hiddenActive[0] = activeNode
    for i in range(1, hiddenLayerSize):
        activeNode = ReLUIng(np.matmul(weightTensor[1][i - 1], hiddenActive[i - 1]) + bias[1][i - 1])
        hiddenActive[i] = activeNode
    outputActive = ReLUIng(np.matmul(weightTensor[2], hiddenActive[-1]) + bias[2])
    predict = np.round(softmax(outputActive), 3)
    return predict, outputActive ,hiddenActive

def trainEach(weightMatrix, biasMatrix, dLoss, dAcFunc):
    dAcFunc[dAcFunc > 0] = 1
    gradient = np.array(weightMatrix)
    gradientBias = np.array(biasMatrix)
    lenght = gradient.shape[1]
    for i in range(lenght):
        weightColumn = gradient[:,i]
        gradientBias[i] = np.sum(dLoss * dAcFunc * 1)
        gradient[0][i] = np.sum(dLoss * dAcFunc * weightColumn)
    return gradient, gradientBias

# Back propagation
# Our loss function is RMSE (y_hat - y) ^ 2
def train(weightTensor, biasTensor, predict, outputActive, hiddenActive, label):
    gradientWeight, gradientBias = initializeNetwork()
    learningRate = 0.01
    lenght = hiddenLayerSize - 2
    # TODO Fix here
    dLoss = 2 * (predict - label)
    dAcFunc = outputActive.copy()
    gradientWeight[2], gradientBias[2] = trainEach(weightTensor[2], biasTensor[2], dLoss, dAcFunc)
    for i in range(lenght, -1, -1): # hidden size actually have two layer of edges
        if i == (lenght):
            dLoss = gradientWeight[2] 
        else:
            dLoss = np.array(gradientWeight[1][i])
        dLoss = dLoss.sum(axis=0)
        dAcFunc = hiddenActive[i]
        gradientWeight[1][i], gradientBias[1][i] = trainEach(weightTensor[1][i], biasTensor[1][i], dLoss, dAcFunc)
    dLoss = np.array(gradientWeight[1][0])
    dLoss = dLoss.sum(axis=0)
    dAcFunc = hiddenActive[i]
    gradientWeight[0], gradientBias[0] = trainEach(weightTensor[0], biasTensor[0], dLoss, dAcFunc)
    # assgin value result
    weightTensor[2] -= gradientWeight[2] * learningRate
    biasTensor[2] -= gradientBias[2] * learningRate
    for i  in range(lenght):
        weightTensor[1][i] -= gradientWeight[1][i] * learningRate
        biasTensor[1][i] -= gradientBias[1][i] * learningRate
    weightTensor[0] -= gradientWeight[0] * learningRate
    biasTensor[0] -= gradientBias[0] * learningRate
    return weightTensor, biasTensor

# MAIN SCRIPT
# get saved bias and weight data from saved data
if (haveSaved == False):
    weightTensor, biasTensor = initializeNetwork()
else:
    # TODO get the saved dat
    pass

predict, outputActive, hiddenActive  = activateNodes(weightTensor, biasTensor, np.array([10, 10, 10]))
weightTensor, biasTensor = train(weightTensor, biasTensor, predict, outputActive, hiddenActive, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
if save:
    print(biasTensor)
print("Predict", predict)