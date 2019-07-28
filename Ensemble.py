from numpy import zeros, argmax, multiply, ones, arange, concatenate, divide, array, eye
from math import floor
class ensemblePredict:
    def __init__(self, models, data, dataLabels):
        self.models = models
        self.data = data
        self.dataLabels = dataLabels

    def predict(self):
        self.predictions = []
        for i in range(len(self.models)):
            self.predictions.append(self.models[i].probPredict(self.data))

    def toMajVot(self):
        self.majVot = []
        # for i in range(len(self.predictions)):
        #     maxIdx = argmax(self.predictions[i], axis=1)
        #     majVotM = zeros(self.predictions[0].shape)
        #     for j in range(len(self.predictions[i][:, 0])):
        #         majVotM[j, maxIdx[j]] = 1

        #     self.majVot.append(majVotM)
        for i in range(len(self.predictions)):
            maxIdx = argmax(self.predictions[i], axis=1)
            maxIdx = array([maxIdx]).reshape(-1)
            majVotM = eye(len(self.predictions[i][0]))[maxIdx]
            self.majVot.append(majVotM)
        #     maxIdx = argmax(self.predictions[i], axis=2)
        #     ordIdx = arange(len(self.predictions[0][:, 0]))
        #     maxIdx = concatenate((ordIdx, maxIdx), )
        #     majVotM = zeros(self.predictions[0].shape)
        # self.majVot[maxIdx] = 1

    def calcMajVot(self, printAcc=False):
        self.majAvg = zeros(self.predictions[0].shape)
        self.toMajVot()
        for i in range(len(self.predictions)):
            self.majAvg += self.predictions[i]

        self.majAvg = self.majAvg / (i+1)
        self.classify(self.majAvg, 'max', True, "Majority Voting")

    def calcProbAvg(self, printAcc=False):
        self.probAvg = zeros(self.predictions[0].shape)
        for i in range(len(self.predictions)):
            self.probAvg += self.predictions[i]

        self.probAvg = self.probAvg / (i+1)

        self.classify(self.probAvg, 'max', True, "Average fusion")
        # self.majVotResults = argmax(self.probAvg, axis = 1) + 1
        # if printAcc:
        #     accuracy = (self.majVotResults - self.dataLabels).tolist().count(0)/len(self.dataLabels)
        #     print("Accuracy: {:.2%}".format(accuracy))

        return self.probAvg

    def calcProbMult(self, printAcc=False):
        self.probMult = ones(self.predictions[0].shape)
        for i in range(len(self.models)):
            self.probMult = multiply(self.probMult, self.predictions[i])

        self.classify(self.probMult, 'max', True, "Multiply fusion")

    def classify(self, probMat, classType, printAcc = False, idText = "Accuracy"):
        if classType == 'max':
            results = argmax(probMat, axis = 1) + 1

        if printAcc:
            accuracy = (results - self.dataLabels).tolist().count(0)/len(self.dataLabels)
            print(idText + ": {:.2%}".format(accuracy))
    