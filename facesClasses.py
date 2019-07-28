import scipy.io
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from numpy import zeros, matmul, transpose, array, float64, uint8, argsort, ones, square, mean
from numpy.linalg import eigh, eig, norm
from numpy import pad

from sklearn.neighbors import KNeighborsClassifier

# Class to display an image from a vector
class faceImg:
    def __init__(self, faceArr, display=False, width=46, height=56):
        # Reshape the vector to a width x height array of points
        self.faceArr = faceArr.reshape(width, height).transpose()
        
        # Create Image object and import data
        self.im = Image.fromarray(self.faceArr)

        if display:
            # Display
            self.show()

    # Function to display the image stored in the class
    def show(self):
        self.im.show()

# Class that implements PCA and face recognition
class faces:
    def __init__(self, trainPer, data = [], filterId = 0, dataPath = '.\\Data\\face.mat', randSeed = 1000001):
        # If the data array is empty, import data from a mat file in dataPath
        if data == []:
            # Import mat file
            importData = scipy.io.loadmat(dataPath)

            # Import faces
            self.faces = importData["X"]
            # Import classes labels
            self.ids = importData["l"][0]

            # If 'filterId' is set, keep only the data for that class label
            if filterId != 0:
                self.faces = self.faces[:,(filterId - 1)*10:filterId*10]
                self.ids = self.ids[(filterId - 1)*10:filterId*10]

            # Make the train and test sets, with the random seed set to 'randseed'
            self.makeTrainTestSets(trainPer, randSeed)

        # If training and test sets are specified, use those
        else:
            self.facesTrain = data[0]
            self.idsTrain = data[1]
            self.facesTest = data[2]
            self.idsTest = data[3]

        # Compute the average training face
        self.avgTrainFace = self.getAverageTrainFace()

        # Change facesTrain format - necessary?
        self.facesTrain = array(self.facesTrain, dtype=float64)

        # Make a normalised array of the training and test data. Each column vector of the arrays is a normalised face
        self.facesTrainNorm = (self.facesTrain.T - self.avgTrainFace.T).T
        self.facesTestNorm = (self.facesTest.T - self.avgTrainFace.T).T
        #self.facesTrainNorm = self.facesTrain - array(matmul(self.avgTrainFace, ones(len(self.facesTrain[0])).reshape(1, len(self.facesTrain[0]))), dtype=float64)
        #self.facesTestNorm = self.facesTest - array(matmul(self.avgTrainFace, ones(len(self.facesTest[0])).reshape(1, len(self.facesTest[0]))), dtype=float64)

    # Function to return a face vector from the train faces dataset.
    def getTrainFace(self, faceNum):
        return self.facesTrain[:, faceNum]

    # Function to return a face vector from the test faces dataset.
    def getTestFace(self, faceNum):
        return self.facesTest[:, faceNum]

    # Function that splits the original faces dataset into a train and test set. The train set contains 'trainPer' of the samples of the entire set
    def makeTrainTestSets(self, trainPer, randSeed):
        # Perform dataset split
        self.facesTrain, self.facesTest, self.idsTrain, self.idsTest = train_test_split(self.faces.T, self.ids, train_size=trainPer, random_state=randSeed, stratify=self.ids)
        self.facesTrain = self.facesTrain.T     # Transpose so that each column is a face
        self.facesTest = self.facesTest.T       # Transpose so that each column is a face

    # Function that returns the average face of the entire dataset
    def getAverageFace(self):
        # Add all faces
        avgFace = zeros(len(self.faces[:,0]))
        for i in range(len(self.faces[0])):
            avgFace = avgFace + self.faces[:,i]
        
        # Divide by the number of faces
        avgFace = avgFace/(i+1)

        return avgFace.reshape(len(avgFace), 1)

    # Function that returns the average face of the training dataset
    def getAverageTrainFace(self):
        # Add all faces
        avgTFace = zeros(len(self.facesTrain[:,0]))
        for i in range(len(self.facesTrain[0])):
            avgTFace = avgTFace + self.facesTrain[:,i]
            
        # Divide by the number of faces
        avgTFace = avgTFace/(i+1)

        return avgTFace.reshape(len(avgTFace), 1)

    # Function that calculates the eigenvalues and eigenvectors of the covarience matrix formed by the training data.
    def calEigenFaces(self, numOfEigs = 0, formula = "ATA"):
        # Compute the covariance matrix
        self.calcCovMat(formula)

        # Calculate the eigenvectors and eigenvalues
        self.calcEig(numOfEigs)

        # If the formula set for the covariance matrix calculation is 'ATA', transform eigenvectors
        if formula == "ATA":
            # Perform transformation
            self.trEigVecs = matmul(self.facesTrainNorm, self.trEigVecs)
            # Normalise for unit magnitude
            self.trEigVecs = preprocessing.normalize(self.trEigVecs, axis=0)

    # Function that calculates the covariance matrix of the training dataset
    def calcCovMat(self, formula = "AAT"):
        if formula == "AAT":
            # Covariance matrix of size 2576 x 2576
            self.trainCovMat = matmul(self.facesTrainNorm, self.facesTrainNorm.T)/len(self.facesTrainNorm[0])
        elif formula == "ATA":
            # Covariance matrix of size equal to the size of the training set (i.e. the number of faces in the training set)
            self.trainCovMat = matmul(self.facesTrainNorm.T, self.facesTrainNorm)/len(self.facesTrainNorm[0])

    # Function that returns the training covariance matrix
    def getCovMat(self):
        return self.trainCovMat

    # Function that calculates the eigenvalues and eigenvectors of the training covariance matrix. The first 'numOfEigs' eigenvectors are kept
    def calcEig(self, numOfEigs):
        # Perform efficient eigenvalues and eigenvectors decomposition (only for symmetric matrices)
        self.trEigVals, self.trEigVecs = eigh(self.trainCovMat)
        self.trEigVals = self.trEigVals[::-1]       # Reverse the order of the eigenvalues (for descending set)
        self.trEigVecs = self.trEigVecs[:,::-1]     # Reverse the order of the columns (for descending set)

        # # Old code - not efficient implementation that requires reordering
        # eig implementation with reordering of eigenvalues and eigenvectors
        # self.trEigVals, self.trEigVecs = eig(self.trainCovMat)
        # self.idx = self.trEigVals.argsort()[::-1]
        # self.trEigVals = self.trEigVals[self.idx]
        # self.trEigVecs = self.trEigVecs[:, self.idx]
        
        # If 'numOfEigs' is 0, keep all the eigenvalues and eigenvectors, otherwise crop the set
        if numOfEigs != 0:
            self.trEigVals = self.trEigVals[0:numOfEigs]        # Keep the first 'numOfEigs' eigenvalues
            self.trEigVecs = self.trEigVecs[:, 0:numOfEigs]     # Keep the first 'numOfEigs' eigenvectors (columns)

    # Function that calculates all the weights corresponding to the training and test faces when projected to the eigenvectors
    def calcFacesWeights(self):
        self.trWeights = matmul(self.facesTrainNorm.T, self.trEigVecs).T    # Training weights, each column of weights corresponds to a column in the training set (i.e. a face). Size: n.Eigenvectors x n.Faces
        self.teWeights = matmul(self.facesTestNorm.T, self.trEigVecs).T     # Test weights, each column of weights corresponds to a column in the test set (i.e. a face). Size: n.Eigenvectors x n.Faces

    # Function that implements kNN learning
    def learnIds(self, nNeighbors):
        # Calculate the
        self.calcFacesWeights()

        self.nn = KNeighborsClassifier(n_neighbors=nNeighbors)
        self.nn.fit(self.trWeights.T, self.idsTrain)

    def predictIds(self):
        self.idsTrPredict = self.nn.predict(self.trWeights.T)
        self.idsTePredict = self.nn.predict(self.teWeights.T)

    def getKNNScores(self):
        trainScore = self.nn.score(self.trWeights.T, self.idsTrain)
        testScore = self.nn.score(self.teWeights.T, self.idsTest)
        return trainScore, testScore

    def confMat(self, dataType):
        if not hasattr(self, 'idsTrPredict') or not hasattr(self, 'idsTePredict'):
            self.predictIds()

        if dataType == "train":
            return confusion_matrix(self.idsTrain, self.idsTrPredict)
        elif dataType == "test":
            return confusion_matrix(self.idsTest, self.idsTePredict)

    def getRecError(self, faceArr):
        #faceArr = faceArr - self.avgTrainFace*ones((1, len(faceArr[0])))
        faceArr = (faceArr.T - self.avgTrainFace.T).T
        weights = matmul(self.trEigVecs.T, faceArr).T
        recon = (matmul(self.trEigVecs, weights.T))#.T + self.avgTrainFace.T).T
        diff = faceArr - recon
        return norm(diff, axis = 0)

    def getMSE(self):
        #faceArr = faceArr - self.avgTrainFace*ones((1, len(faceArr[0])))
        weightsTrain = matmul(self.trEigVecs.T, self.facesTrainNorm).T
        weightsTest = matmul(self.trEigVecs.T, self.facesTestNorm).T

        reconTrain = matmul(self.trEigVecs, weightsTrain.T)
        reconTrain = (reconTrain.T + self.avgTrainFace.T).T

        reconTest = matmul(self.trEigVecs, weightsTest.T)
        reconTest = (reconTest.T + self.avgTrainFace.T).T

        diffTrain = reconTrain - self.facesTrain
        diffTrain = square(diffTrain)

        diffTest = reconTest - self.facesTest
        diffTest = square(diffTest)
        return mean(mean(diffTrain)), mean(mean(diffTest))
