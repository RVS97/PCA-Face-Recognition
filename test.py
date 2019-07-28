import scipy.io
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from numpy import zeros, matmul, transpose, array, float64, uint8, argsort, ones
from numpy.linalg import eigh, eig, norm
from numpy import pad

from sklearn.neighbors import KNeighborsClassifier

import facesClasses as fC

dataPath = '.\\Data\\face.mat'

importData = scipy.io.loadmat(dataPath)
faces = importData["X"]
ids = importData["l"][0]

faces = faces[:, 0:10]
ids = ids[0:10]

face = faces[:, 0]
avgFace = zeros(face.shape)
for i in range(len(faces[0])):
    avgFace = avgFace + faces[:, i]

avgFace = avgFace/(i+1)
facesNorm = (faces.T - avgFace.T).T

S = matmul(facesNorm, facesNorm.T)/len(facesNorm[0])

vals, vecs = eigh(S)
vals = vals[::-1]
vecs = vecs[:, ::-1]

vecs = vecs[:, 0:10]
w = matmul(face, vecs)
rec = matmul(vecs, w)

img = fC.faceImg(matmul(matmul(vecs.T, vecs), face))
img.show()