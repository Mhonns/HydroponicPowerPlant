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
haveSaved = True       # true to use the weight and bias the saved data
save = False            # admit to save the weight and bias or not
getData = False         # use the input data or not
epoch = 5               # epoch

# assign all weights and bias
def initializeNetwork():
    weightTensor = np.array(
        [np.random.uniform(low=-1,high=1,size=(intputSize, hiddenSize)),
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
    activeNode = ReLUIng(np.matmul(inputData,weightTensor[0]) + bias[0])
    hiddenActive[0] = activeNode
    for i in range(1, hiddenLayerSize):
        activeNode = ReLUIng(np.matmul( hiddenActive[i - 1],weightTensor[1][i - 1]) + bias[1][i - 1])
        hiddenActive[i] = activeNode
    outputActive = (np.matmul(hiddenActive[-1],weightTensor[2]) + bias[2])
    predict = outputActive
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
    # grad: (1 x dim)
    # print(grad.shape)
    # print(activation.shape)
    return transposeArray(activation)@grad
def dLinear_b(activation,weight,bias,grad):
    b_shape = bias.shape[0]
    return grad.reshape(b_shape)
def dLinear_activation(activation,weight,bias,grad):
    return grad@(weight.T)

# Back propagation
# Our loss function is RMSE (y_hat - y) ^ 2
# Args
#   weightTensor - current weight
#   biasTensor - current bias
#   activeNodes - list contain [hiddenActive, outputActive]
#   predict - guessed value
#   label   - answer value
def backProp(weightTensor, biasTensor, inputData, activeNodes, predict, label):
    learningRate = 0.001
    lenght = hiddenLayerSize - 2
    # Last Layer
    dLoss_dpredict = 2 * (predict - label) 
    dLoss_do = (predict * dLoss_dpredict).reshape(1,1) # (1,1)
    dLoss_dWeight = dLinear_W(activeNodes[-1],weightTensor[2],biasTensor[2],dLoss_do) # 1x4
    dLoss_dBias = dLinear_b(activeNodes[-1],weightTensor[2],biasTensor[2],dLoss_do) # 1x1
    dLoss_dactivation = dLinear_activation(activeNodes[-1],weightTensor[2],biasTensor[2],dLoss_do) # 1x4
    weightTensor[2] -= dLoss_dWeight * learningRate
    biasTensor[2]   -= dLoss_dBias * learningRate
    # After first layers
    for n in range(lenght, -1, -1):
        dLoss_dprerelu = dActivation(activeNodes[n + 1],dLoss_dactivation) # 1x4
        dLoss_dWN = dLinear_W(activeNodes[n], weightTensor[1][n], biasTensor[1][n], dLoss_dprerelu) # 4x4 (dLoss_dprerelu.T)@()
        dLoss_dBN = dLinear_b(activeNodes[n], weightTensor[1][n], biasTensor[1][n], dLoss_dprerelu)
        dLoss_dactivation = dLinear_activation(activeNodes[n],weightTensor[1][n],biasTensor[1][n],dLoss_dprerelu)
        #dLoss_dactivation = dMatrix_activation(activeNodes[n],weightTensor[1][n],biasTensor[1][n],dLoss_dprerelu) # 1x4
        weightTensor[1][n] -= dLoss_dWN * learningRate
        biasTensor[1][n] -= dLoss_dBN * learningRate
    # First Layer
    dLoss_dprerelu = dActivation(activeNodes[0],dLoss_dactivation)
    dLoss_dWN = dLinear_W(inputData, weightTensor[0], biasTensor[0], dLoss_dprerelu)
    dLoss_dBN = dLinear_b(inputData, weightTensor[0], biasTensor[0], dLoss_dprerelu)
    weightTensor[0] -= dLoss_dWN * learningRate
    biasTensor[0] -= dLoss_dBN * learningRate
    return weightTensor, biasTensor

# MAIN SCRIPT
# get saved bias and weight data from saved data
weightTensor, biasTensor = initializeNetwork()
if (haveSaved == True):
    for i in range(3):
        tempW = np.genfromtxt("SaveSlot_0/weight"+str(i),delimiter=',')
        tempB = np.genfromtxt("SaveSlot_0/bias"+str(i),delimiter=',')
        if i == 0:
            weightTensor[i] = tempW.reshape(intputSize, hiddenSize)
            biasTensor[i]   = tempB
        elif i == 1:
            weightTensor[i] = tempW.reshape(hiddenLayerSize - 1, hiddenSize, hiddenSize)
            biasTensor[i]   = tempB.reshape(hiddenLayerSize - 1, hiddenSize)
        else:
            weightTensor[i] = tempW.reshape(hiddenSize, outputSize)
            biasTensor[i] = tempB.reshape(1)


# get the test set
#   make up data
label = 1
inputData = np.array([10, 10, 10]).reshape(1,3) # bz x hiddenSize
if getData == True:
    pass
    
    

predict, activeNodes = forwardProp(weightTensor, biasTensor, inputData)
for i in range(epoch):
    weightTensor, biasTensor = backProp(weightTensor, biasTensor, inputData, activeNodes, predict, label)
    
# print debug
predict, activeNodes = forwardProp(weightTensor, biasTensor, inputData)
print("\nPredict", predict)

if save:
    for i in range(3):
        weightTensor[i].tofile("SaveSlot_0/weight"+str(i), sep=',', format='%s')
        biasTensor[i].tofile("SaveSlot_0/bias"+str(i), sep=',', format='%s')
