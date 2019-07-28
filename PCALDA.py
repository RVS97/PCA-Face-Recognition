import sys
import facesClasses as fC
import plotFunc
from matplotlib import pyplot
from numpy import matmul, concatenate, array, where, zeros, linalg
import reportFuncs as rF

from sklearn.neighbors import KNeighborsClassifier
## s ####################################################

class PCALDA:
    def __init__(self, faces):
        self.allFaces = faces
        self.avgTFace = self.allFaces.avgTrainFace

        # Extract the training and test sets of the allFaces class
        self.facesTrain = faces.facesTrain
        self.idsTrain = faces.idsTrain
        self.facesTest = faces.facesTest
        self.idsTest = faces.idsTest

        self.facesTrainNorm = (self.facesTrain.T - self.avgTFace.T).T
        self.facesTestNorm = (self.facesTest.T - self.avgTFace.T).T

        self.nClasses = len(set(self.idsTrain))
        self.faceSize = len(self.avgTFace)
        
    def calcW_PCA(self, M_PCA):
        self.allFaces.calEigenFaces()
        self.W_PCA = self.allFaces.trEigVecs[:, 0:M_PCA]

    def setW_PCA(self, W_PCA):
        self.W_PCA = W_PCA

    def calcScatterMatrices(self):
        self.facesArr = []

        self.avgCFace = zeros((len(self.facesTrain[:,0]), self.nClasses))


        for i in range(self.nClasses):
            trainI = where(self.idsTrain == i+1)[0]
            testI = where(self.idsTest == i+1)[0]

            self.facesArr.append(fC.faces(0.7, [self.facesTrain[:,trainI], self.idsTrain[trainI], self.facesTest[:,testI], self.idsTest[testI]]))
            #fC.faceImg(facesArr[i].avgTrainFace[0], True)
            self.avgCFace[:, i] = self.facesArr[i].avgTrainFace[:, 0]
            #facesArr[i].calEigenFaces(4,"ATA")
            
        self.normAvgCFace = (self.avgCFace.T - self.avgTFace.T).T

        self.S_B = zeros((len(self.avgTFace), len(self.avgTFace)))
        self.S_W = zeros((len(self.avgTFace), len(self.avgTFace)))

        self.S_B = matmul(self.normAvgCFace, self.normAvgCFace.T)

        for i in range(self.nClasses):
            #self.S_B = self.S_B + matmul(array([self.normAvgCFace[:, i]]).T, array([self.normAvgCFace[:, i]]))

            trainI = where(self.idsTrain == i+1)[0]
            #testI = where(self.idsTest == i+1)[0]
            c = (self.facesTrainNorm[:, trainI].T - self.avgCFace[:, i].T).T
            self.S_W = self.S_W + matmul(c, c.T)
            #for j in range(len(trainI)):
            #    diffV = self.facesTrainNorm[:, trainI[j]] - self.avgCFace[:, i]
            #    self.S_W = self.S_W + matmul(array([diffV]).T, array([diffV]))

    def calcPCALDA(self, M_LDA):
        self.PS_B = matmul(matmul(self.W_PCA.T, self.S_B), self.W_PCA)
        self.PS_W = matmul(matmul(self.W_PCA.T, self.S_W), self.W_PCA)
        self.vals, self.vecs = linalg.eig(matmul(linalg.inv(self.PS_W), self.PS_B))
        idx = self.vals.argsort()[::-1]
        self.vals = self.vals[idx]
        self.vecs = self.vecs[:, idx]
        self.vals = self.vals[0:M_LDA]
        self.vecs = self.vecs[:, 0:M_LDA]
        self.W_OPT = matmul(self.W_PCA, self.vecs.real)

    def classify(self, nNeighbors, printScore = False):
        self.trWeights = matmul(self.facesTrainNorm.T, self.W_OPT).T
        self.teWeights = matmul(self.facesTestNorm.T, self.W_OPT).T

        self.nn = KNeighborsClassifier(n_neighbors=nNeighbors)
        self.nn.fit(self.trWeights.T, self.idsTrain)
        self.trainScore = self.nn.score(self.trWeights.T, self.idsTrain)
        self.testScore = self.nn.score(self.teWeights.T, self.idsTest)

        if printScore:
            print("Train accuracy: {:.2%}, test score: {:.2%}".format(self.trainScore, self.testScore))

    def probPredict(self, data):
        dataNorm = data - self.avgTFace
        dataWeights = matmul(dataNorm.T, self.W_OPT)
        # data faces arranged by columns
        return self.nn.predict_proba(dataWeights)