import math
import random
import pickle
import os

class NeuralLayer:
    def __init__(self, numberOfInputs, numberOfOutputs, activationFunction, weights = None):
        self.Weights = []
        self.NumberOfInputs = numberOfInputs
        self.NumberOfNeurons = numberOfOutputs
        self.ActivationFunction = activationFunction
        if weights == None:
            for i in range(self.NumberOfNeurons):
                self.Weights.append([random.uniform(-1,1) for j in range(self.NumberOfInputs + 1)])
        else:
            for i in range(self.NumberOfNeurons):
                self.Weights.append(weights[i])
    def Calculate(self, inputs):
        tempInputs = inputs.copy()
        tempInputs.insert(0,1.0)
        outputs = []
        for i in range(self.NumberOfNeurons):
            sum = 0
            for j in range(len(self.Weights[i])):
                sum += tempInputs[j] * self.Weights[i][j]
            outputs.append(self.ActivationFunction(sum))
        return outputs
class TanhTeachingNeuralLayer(NeuralLayer):
    def __init__(self, sourceNeuralLayer):
        super().__init__(sourceNeuralLayer.NumberOfInputs, sourceNeuralLayer.NumberOfNeurons, sourceNeuralLayer.ActivationFunction, sourceNeuralLayer.Weights)
        self.Inputs = [0.0 for i in range(sourceNeuralLayer.NumberOfInputs)]
        self.Outputs = [0.0 for i in range(sourceNeuralLayer.NumberOfNeurons)]
        self.Errors = [0.0 for i in range(sourceNeuralLayer.NumberOfNeurons)]
        self.DeltaWeights = [[0.0 for j in range(len(sourceNeuralLayer.Weights[i]))] for i in range(len(sourceNeuralLayer.Weights))]
    def Calculate(self, inputs):
        self.Inputs = inputs
        self.Outputs = super().Calculate(inputs)
        return self.Outputs
    def CalculateDeltaWeights(self):
        self.DeltaWeights = [[0.0 for j in range(len(self.Weights[i]))] for i in range(len(self.Weights))]
        for i in range(len(self.Weights)):
            for j in range(len(self.Weights[i])):
                if j == 0:
                    self.DeltaWeights[i][j] = 1 * (1 - self.Outputs[i]) * self.Errors[i]
                else:
                    self.DeltaWeights[i][j] = self.Inputs[j-1] * (1 - self.Outputs[i]) * self.Errors[i]
        return self.DeltaWeights
    def BackPropgateError(self, errors):
        self.Errors = errors
        backLayerError = [0.0 for i in range(self.NumberOfInputs)]
        for i in range(len(self.Weights)):
            sum = 0
            for j in range(len(self.Weights[i])):
                sum += abs(self.Weights[i][j])
            for j in range(1,len(self.Weights[i])):
                backLayerError[j-1] += self.Errors[i] * (self.Weights[i][j]/sum)
        return backLayerError

class NeuralNetwork:
    def __init__(self, map, activationFunction, weights = None):
        self.NeuronLayers = []
        self.Map = map
        if weights == None:
            weights = [None for i in range(len(map) - 1)]
        for i in range(len(map) - 1):
            self.NeuronLayers.append(NeuralLayer(map[i],map[i+1],activationFunction,weights[i]))
    def Calculate(self, inputs):
        temp = inputs
        for i in range(len(self.NeuronLayers)):
            temp = self.NeuronLayers[i].Calculate(temp)
        return temp

class NeuralTeacherTanh:
    def __init__(self, neuralNetwork, learningRate):
        self.NeuralNetwork = neuralNetwork
        self.LearningRate = learningRate
        self.TeachingNeuronLayers = [TanhTeachingNeuralLayer(nL) for nL in self.NeuralNetwork.NeuronLayers]
    def Teach(self, inputs, requiredOutputs):
        numberOfLayers = len(self.TeachingNeuronLayers)
        numberOfOutputs = self.TeachingNeuronLayers[numberOfLayers-1].NumberOfNeurons
        #Feed Forward
        self.TeachingNeuronLayers[0].Calculate(inputs)
        for i in range(1,numberOfLayers):
            self.TeachingNeuronLayers[i].Calculate(self.TeachingNeuronLayers[i-1].Outputs)
        #Calculate all output errors
        tempErrors = [requiredOutputs[i] - self.TeachingNeuronLayers[numberOfLayers-1].Outputs[i] for i in range(numberOfOutputs)]
        for i in range(numberOfLayers-1,0,-1):
            tempErrors = self.TeachingNeuronLayers[i].BackPropgateError(tempErrors)

        for nL in self.TeachingNeuronLayers:
            nL.CalculateDeltaWeights()
        for i in range(numberOfLayers):
            for j in range(self.NeuralNetwork.Map[i+1]):
                for w in range(len(self.NeuralNetwork.NeuronLayers[i].Weights[j])):
                    self.NeuralNetwork.NeuronLayers[i].Weights[j][w] += self.TeachingNeuronLayers[i].DeltaWeights[j][w] * self.LearningRate
        return self.NeuralNetwork
def aFunc(x):
    return math.tanh(x)
def Normalize(fromMin, fromMax, toMin, toMax, value):
    return (((value - fromMin)/(fromMax - fromMin))*(toMax - toMin))+ toMin
def SaveNeuralNetwork(neuralNetwork,path):
    with open(path, 'wb') as f:
        pickle.dump(neuralNetwork,f)
def LoadNeuralNetwork(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

scriptDirectory = os.path.dirname(os.path.realpath(__file__))
print(scriptDirectory)
name = input('Enter saved neural network path or enter None:')
if name == 'None':
    nn = NeuralNetwork([784,112,10],aFunc)
else:
    nn = LoadNeuralNetwork(scriptDirectory + '\\' +  name)
teacher = NeuralTeacherTanh(nn,0.01)
labelFile = open(scriptDirectory + '\\' + 'lables.idx1-ubyte','rb')
imageFile = open(scriptDirectory + '\\' +  'images.idx3-ubyte','rb')
trainingData = []
try:
    for i in range(1000):
        inputs = []
        requiredOutputs = []
        lable = int.from_bytes(labelFile.read(1),byteorder='big')
        for j in range(10):
            if j == lable:
                requiredOutputs.append(1.0)
            else:
                requiredOutputs.append(-1.0)
        for j in range(784):
            pixel = int.from_bytes(imageFile.read(1),byteorder='big')
            inputs.append(Normalize(0,255,-1,1,pixel))
        trainingData.append((lable,inputs,requiredOutputs))


finally:
    labelFile.close()
    imageFile.close()
print('Done loading')
print('Now training...')
setNumber = 0
for i in range(0):
    r = random.randint(0,len(trainingData)-1)
    set = trainingData[r]
    nn = teacher.Teach(set[1],set[2])
    setNumber += 1
    if setNumber%100 == 0:
        print(setNumber)
SaveNeuralNetwork(nn,scriptDirectory + '\\' +  'Digits')
print('Done training, you can now test')

while True:
    r = random.randint(0,len(trainingData)-1)
    print(trainingData[r][0])
    print(trainingData[r][2])
    for j in range(28):
        for k in range(28):
            if trainingData[r][1][28*j+k] >-0.8:
                print('0',end='')
            else:
                print(' ',end='')

        print()
    outputs = nn.Calculate(trainingData[r][1])
    print(outputs)
    max = -2
    lable = 0
    for j in range(len(outputs)):
        if outputs[j] > max:
            max = outputs[j]
            lable = j
    print(lable)
    if lable == trainingData[r][0]:
        print('Correct')
    else:
        print('Incorrect')
    input()
