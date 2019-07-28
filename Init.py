import sys
import facesClasses as fC
import plotFunc
from matplotlib import pyplot
from numpy import matmul, concatenate, array, where, zeros
import reportFuncs as rF
import PCALDA as PCALDA_Class
import random
import Ensemble
from math import floor
import pickle
import timeit

# EXECUTION CODE FOR ALL DIFFERENT IMPLEMENTATIONS
# UNCOMMENT/COMMENT EACH SECTION ACOORDINGLY FOR USE


########## Normal face recognition with PCA ####################################################

# Instantiate the faces class that contains all functionality for PCA decomposition and kNN, and partition data into training and test set
allFaces = fC.faces(trainPer=0.7)

# Calculate the eigenfaces, keep the first 'numOfEigs' eigenvalues, and perform the calculation in the order specified by 'formula'
allFaces.calEigenFaces(numOfEigs=150, formula="ATA")

Plot: Create eigenvalues plot
rF.plotEigvals(allFaces)

# Plot: Display eigenfaces
rF.plotEigFaces(allFaces)

rF.compPlot(allFaces)

# Plot: Create plot with reconstructed faces, each with a different number of bases
rF.plotReconFaces(allFaces)

rF.kNNSuccFail(allFaces)
# Create and fit kNN model 
#for i in [1, 3, 5, 7]:
allFaces.learnIds(nNeighbors=1)
# Get test and train accuracies for varying number of eigenvalues
for i in range(5):
    allFaces = fC.faces(trainPer=0.7, randSeed=10101010%502302) 
 rF.getAcc(allFaces, range(363))

# Calculate model accuracy
trainScore, testScore = allFaces.getKNNScores()
print("Train accuracy: {:.2%}, test score: {:.2%}".format(trainScore, testScore))

# Plot train confusion matrix
pyplot.figure()
plotFunc.plot_confusion_matrix(allFaces.confMat("train"), range(1, max(allFaces.idsTrain)+1), title="NN Confusion Matrix - Train")

# Plot test confusion matrix
pyplot.figure()
plotFunc.plot_confusion_matrix(allFaces.confMat("test"), range(1, max(allFaces.idsTest)+1), title="NN Confusion Matrix - Test")
pyplot.show()

# Exit
sys.exit(0)

########## Face recognition using reconstruction errors ####################################################

# # Instanciate the faces class that contains all functionality for PCA decomposition and kNN, and partition data into training and test set
# # This is done to preserve the exact training and test sets as in the normal implementation of face recognition
# allFaces = fC.faces(0.7)

# # Extract the training and test sets of the allFaces class
# facesTrain = allFaces.facesTrain
# idsTrain = allFaces.idsTrain
# facesTest = allFaces.facesTest
# idsTest = allFaces.idsTest


# eMatTrain = zeros((52, len(facesTrain[0])))
# eMatTest = zeros((52, len(facesTest[0])))

# ##trainAcc = zeros(6)
# ##testAcc = zeros(6)

# ##for j in range(6):
# ##   print(j)
# facesArr = []

# for i in range(52):
#     trainI = where(idsTrain == i+1)[0]
#     testI = where(idsTest == i+1)[0]

#     facesArr.append(fC.faces(0.7, [facesTrain[:,trainI], idsTrain[trainI], facesTest[:,testI], idsTest[testI]]))
#     facesArr[i].calEigenFaces(j + 1,"ATA")
    
#     eMatTrain[i, :] = facesArr[i].getRecError(facesTrain)
#     eMatTest[i, :] = facesArr[i].getRecError(facesTest)

# eTrain = eMatTrain.argmin(axis=0) + 1
# eTest = eMatTest.argmin(axis=0) + 1

# trAcc = (idsTrain - eTrain).tolist().count(0)/len(idsTrain)
# teAcc = (idsTest - eTest).tolist().count(0)/len(idsTest)

# ##   trainAcc[j] = trAcc
# ##   testAcc[j] = teAcc

# ##with open('RecErrAcc.pkl', 'wb') as f:
# ##    pickle.dump([trainAcc, testAcc], f)

# print(trAcc)
# print(teAcc)

# # Exit
# sys.exit(0)

########## Face recognition using PCA-LDA ####################################################
#84 % accuracy
# allFaces = fC.faces(0.7)

# PCALDA = PCALDA_Class.PCALDA(allFaces)
# PCALDA.calcW_PCA(150)
# PCALDA.calcScatterMatrices()
# PCALDA.calcPCALDA(51)
# PCALDA.classify(1, True)
# # Exit
# sys.exit(0)

########## Face recognition using PCA-LDA Bagging ####################################################
# allFaces = fC.faces(0.7)
# allFaces.calEigenFaces()
# W_PCA = allFaces.trEigVecs

# # Bagging
# random.seed(8344)   # Set random seed
# nSets = 10


# trainIdx = allFaces.idsTrain.argsort()
# ordFacesTrainSet = allFaces.facesTrain[:, trainIdx]
# ordIdsTrainSet = allFaces.idsTrain[trainIdx]
# classLength = ordIdsTrainSet.tolist().count(1)
# nClasses = len(set(ordIdsTrainSet))

# setSize = len(allFaces.facesTrain[0])
# classSamplesPerSet = floor(setSize/nClasses)
# setSize = classSamplesPerSet*nClasses


# randFacesTrainSets = []
# randIdsTrainSets = []
# for i in range(nSets):
#     facesTrain = zeros((len(allFaces.facesTrain[:, 0]), setSize))
#     idsTrain = zeros(setSize)

#     for j in range(nClasses):

#         for k in range(classSamplesPerSet):
#             idx = random.randint(0, classLength-1)
#             facesTrain[:, (j-1)*classSamplesPerSet + k] = ordFacesTrainSet[:, (j-1)*classLength + idx]
#             idsTrain[(j-1)*classSamplesPerSet + k] = ordIdsTrainSet[(j-1)*classLength + idx]

#     randFacesTrainSets.append(facesTrain)
#     randIdsTrainSets.append(idsTrain)

# models = []
# for i in range(nSets):
#     models.append(PCALDA_Class.PCALDA(fC.faces(0.7, [randFacesTrainSets[i], randIdsTrainSets[i], allFaces.facesTest, allFaces.idsTest])))
#     models[i].calcW_PCA(150)
#     #PCALDA.setW_PCA(W_PCA)
#     models[i].calcScatterMatrices()
#     models[i].calcPCALDA(51)
#     models[i].classify(1, True)

# ensemble = Ensemble.ensemblePredict(models, allFaces.facesTest, allFaces.idsTest)
# ensemble.predict()
# ensemble.calcProbAvg(True)

# # Exit
# sys.exit(0)

########## Face recognition using PCA-LDA Feature rand ####################################################
# allFaces = fC.faces(0.7)
# allFaces.calEigenFaces()
# W_PCA = allFaces.trEigVecs

# # Bagging
# random.seed(8344)   # Set random seed
# nSets = 10

# W_PCASets = []
# M0 = 50
# M1 = 100
# MPCA = M0 + M1

# for i in range(nSets):
#     RandW_PCA = zeros((len(W_PCA[:, 0]), MPCA))
#     RandW_PCA[:, 0:M0] = W_PCA[:, 0:M0]
#     randIdx = random.sample(range(M0, len(W_PCA[0])), M1)
#     RandW_PCA[:, M0:MPCA] = W_PCA[:, randIdx]
#     W_PCASets.append(RandW_PCA)

# models = []
# for i in range(nSets):
#     models.append(PCALDA_Class.PCALDA(fC.faces(0.7, [allFaces.facesTrain, allFaces.idsTrain, allFaces.facesTest, allFaces.idsTest])))
#     models[i].setW_PCA(W_PCASets[i])
#     models[i].calcScatterMatrices()
#     models[i].calcPCALDA(51)
#     models[i].classify(1, True)

# ensemble = Ensemble.ensemblePredict(models, allFaces.facesTest, allFaces.idsTest)
# ensemble.predict()
# ensemble.calcProbAvg(True)

# # Exit
# sys.exit(0)

########## Face recognition using PCA-LDA Ensemble ####################################################
# allFaces = fC.faces(0.7)
# allFaces.calEigenFaces()
# W_PCA = allFaces.trEigVecs

# # Bagging
# random.seed(8344)   # Set random seed
# nSets = 3


# trainIdx = allFaces.idsTrain.argsort()
# ordFacesTrainSet = allFaces.facesTrain[:, trainIdx]
# ordIdsTrainSet = allFaces.idsTrain[trainIdx]
# classLength = ordIdsTrainSet.tolist().count(1)
# nClasses = len(set(ordIdsTrainSet))

# setSize = len(allFaces.facesTrain[0])
# classSamplesPerSet = floor(setSize/nClasses)
# setSize = classSamplesPerSet*nClasses


# randFacesTrainSets = []
# randIdsTrainSets = []
# for i in range(nSets):
#     facesTrain = zeros((len(allFaces.facesTrain[:, 0]), setSize))
#     idsTrain = zeros(setSize)

#     for j in range(nClasses):

#         for k in range(classSamplesPerSet):
#             idx = random.randint(0, classLength-1)
#             facesTrain[:, (j-1)*classSamplesPerSet + k] = ordFacesTrainSet[:, (j-1)*classLength + idx]
#             idsTrain[(j-1)*classSamplesPerSet + k] = ordIdsTrainSet[(j-1)*classLength + idx]

#     randFacesTrainSets.append(facesTrain)
#     randIdsTrainSets.append(idsTrain)

# models = []
# for i in range(nSets):
#     if (i) % 10 == 0:
#         print("Step {:}".format(i+1))
#     models.append(PCALDA_Class.PCALDA(fC.faces(0.7, [randFacesTrainSets[i], randIdsTrainSets[i], allFaces.facesTest, allFaces.idsTest])))
#     models[i].calcW_PCA(150)
#     #PCALDA.setW_PCA(W_PCA)
#     models[i].calcScatterMatrices()
#     models[i].calcPCALDA(51)
#     models[i].classify(1, True)

    
# # Random feature space
# random.seed(8344)   # Set random seed
# nSets2 = 3

# W_PCASets = []
# M0 = 50
# M1 = 100
# MPCA = M0 + M1

# for i in range(nSets2):
#     RandW_PCA = zeros((len(W_PCA[:, 0]), MPCA))
#     RandW_PCA[:, 0:M0] = W_PCA[:, 0:M0]
#     randIdx = random.sample(range(M0, len(W_PCA[0])), M1)
#     RandW_PCA[:, M0:MPCA] = W_PCA[:, randIdx]
#     W_PCASets.append(RandW_PCA)

# for i in range(nSets2):
#     if (i) % 10 == 0:
#         print("Step {:}".format(nSets + i+1))
#     models.append(PCALDA_Class.PCALDA(fC.faces(0.7, [allFaces.facesTrain, allFaces.idsTrain, allFaces.facesTest, allFaces.idsTest])))
#     models[nSets + i].setW_PCA(W_PCASets[i])
#     models[nSets + i].calcScatterMatrices()
#     models[nSets + i].calcPCALDA(51)
#     models[nSets + i].classify(5, True)

# ensemble = Ensemble.ensemblePredict(models, allFaces.facesTest, allFaces.idsTest)
# ensemble.predict()
# ensemble.calcProbAvg(True)
# ensemble.calcMajVot(True)
# ensemble.calcProbMult(True)

# # Exit
# sys.exit()