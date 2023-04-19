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
outputSize = 1          # the number of output nodes (px per day)
hiddenSize = 4          # the number of nodes in hiddenlayers
hiddenLayerSize = 3     # the number of the hidden layers
haveSaved = False       # true to use the weight and bias from weightData.txt
save = True             # admit to save the weight and bias or not
epoch = 1               # epoch

# assign all weights and bias
def initializeNetwork():
    weightTensor = np.array(
        [np.random.uniform(low=-1,high=1,size=(hiddenSize, intputSize)),
        np.array([np.random.uniform(low=-1,high=1,size=(hiddenSize, hiddenSize)) for _ in range(hiddenLayerSize - 1)]),
        np.random.uniform(low=-1,high=1,size=(hiddenSize, outputSize))],
        dtype=object
    )
    biasTensor = np.array(
        [np.ones(hiddenSize),
        np.array([np.ones(hiddenSize) for _ in range(hiddenLayerSize - 1)]),
        np.ones(outputSize)],
        dtype=object
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
def forwardProp(weightTensor, bias, inputData):
    hiddenActive = np.zeros((hiddenLayerSize, hiddenSize), dtype=float)
    outputActive = np.zeros((outputSize), dtype=float)
    activeNode = ReLUIng(np.matmul(weightTensor[0], inputData) + bias[0])
    hiddenActive[0] = activeNode
    for i in range(1, hiddenLayerSize):
        activeNode = ReLUIng(np.matmul(weightTensor[1][i - 1], hiddenActive[i - 1]) + bias[1][i - 1])
        hiddenActive[i] = activeNode
    outputActive = (np.matmul(np.transpose(weightTensor[2]), hiddenActive[-1]) + bias[2])
    predict = np.round(ReLUIng(outputActive), 3)
    return predict, hiddenActive

def transposeArray(a):
    temp = np.copy(a)
    temp = np.transpose(np.matrix(a))
    return temp

# derivative of the ReLU function
def dActivation(z,grad):
    temp = np.copy(z)
    temp[temp < 0] = 0
    temp[temp > 0] = 1
    return temp*grad
def dLinear_W(activation,weight,bias,grad):
    return transposeArray(activation)*grad
def dLinear_b(activation,weight,bias,grad):
    return grad
def dLinear_activation(activation,weight,bias,grad):
    return np.transpose(weight)[0]*grad
def dMatrix_activation(activation,weight,bias,grad):
    tempWeight = np.sum(weight, axis=1)
    return np.transpose(tempWeight)[0] * grad

# Back propagation
# Our loss function is RMSE (y_hat - y) ^ 2
# Args
#   weightTensor - current weight
#   biasTensor - current bias
#   activeNodes - list contain [hiddenActive, outputActive]
#   predict - guessed value
#   label   - answer value
def backProp(weightTensor, biasTensor, inputData, activeNodes, predict, label):
    learningRate = 0.01
    lenght = hiddenLayerSize - 2
    # Last Layer
    dLoss_dpredict = 2 * (predict - label) 
    dLoss_do = dActivation(predict,dLoss_dpredict)
    dLoss_dWeight = dLinear_W(activeNodes[-1],weightTensor[2],biasTensor[2],dLoss_do)
    dLoss_dBias = dLinear_b(activeNodes[-1],weightTensor[2],biasTensor[2],dLoss_do)
    dLoss_dactivation = dLinear_activation(activeNodes[-1],weightTensor[2],biasTensor[2],dLoss_do)
    weightTensor[2] -= dLoss_dWeight * learningRate
    biasTensor[2]   -= dLoss_dBias * learningRate
    # After first layers
    for n in range(lenght, -1, -1):
        dLoss_dprerelu = dActivation(activeNodes[n + 1],dLoss_dactivation)
        dLoss_dWN = dLinear_W(activeNodes[n], weightTensor[1][n], biasTensor[1][n], dLoss_dprerelu)
        dLoss_dBN = dLinear_b(activeNodes[n], weightTensor[1][n], biasTensor[1][n], dLoss_dprerelu)
        dLoss_dactivation = dMatrix_activation(activeNodes[n],weightTensor[1][n],biasTensor[1][n],dLoss_dprerelu)
        weightTensor[1][n] -= dLoss_dWN * learningRate
        biasTensor[1][n] -= dLoss_dBN[0] * learningRate
    # First Layer
    dLoss_dprerelu = dActivation(activeNodes[0],dLoss_dactivation)
    dLoss_dWN = dLinear_W(inputData, weightTensor[0], biasTensor[0], dLoss_dprerelu)
    dLoss_dBN = dLinear_b(inputData, weightTensor[0], biasTensor[0], dLoss_dprerelu)
    weightTensor[0] -= np.transpose(dLoss_dWN) * learningRate
    biasTensor[0] -= dLoss_dBN * learningRate
    return weightTensor, biasTensor

# MAIN SCRIPT
# get saved bias and weight data from saved data
if (haveSaved == False):
    weightTensor, biasTensor = initializeNetwork()
else:
    # TODO get the saved data
    pass

# Test set
label = 1
inputData = np.array([10, 10, 10])

predict, activeNodes = forwardProp(weightTensor, biasTensor, inputData)
print("\nPredict0", predict)
for i in range(epoch):
    weightTensor, biasTensor = backProp(weightTensor, biasTensor, inputData, activeNodes, predict, label)
predict, activeNodes = forwardProp(weightTensor, biasTensor, inputData)

if save:
    pass

print("\nPredict1", predict)