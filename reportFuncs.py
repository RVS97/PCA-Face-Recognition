import facesClasses as fC
import plotFunc
from matplotlib import pyplot
from numpy import matmul, divide, log, zeros, linspace
import pickle

plotFont ={'fontname':'Times New Roman'}
def plotEigvals(allFaces):
    #Eigen values plot
    ###pyplot.plot(abs(allFaces.getEigVals()))
    eigVals = abs(allFaces.trEigVals.real)
    eigVals = eigVals[eigVals.argsort()[::-1]]
    pyplot.rcParams["font.family"] = 'Times New Roman'
    pyplot.semilogy(range(1, len(eigVals)+1), abs(eigVals))
    pyplot.xlabel("Eigenvalue index", fontsize=18)
    pyplot.ylabel("Eigenvalue magnitude", fontsize=18)
    pyplot.title("Eigenvalues of the Covariance Matrix S", fontsize=20)
    pyplot.yticks(fontsize=12)
    pyplot.xticks([1] + [363] + list(range(1000, 3000, 500)), fontsize=12)
    line = pyplot.axvline(x=363)
    line.set_color('black')
    pyplot.grid()
    pyplot.subplots_adjust(top=0.875, bottom=0.15, right=0.955, left=0.14, hspace=0.2)
    pyplot.show()

def plotEigFaces(allFaces):
    pyplot.rcParams["font.family"] = 'Times New Roman'
    eigFaces = allFaces.trEigVecs

    fig=pyplot.figure(figsize=(5, 3.2), dpi=100)
    fig.suptitle("Eigenfaces from the training data", fontsize=18)
    fig.tight_layout()

    ax = pyplot.subplot(2, 5, 1)
    pyplot.imshow(eigFaces[:, 0].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 1')

    ax = pyplot.subplot(2, 5, 2)
    pyplot.imshow(eigFaces[:, 1].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 2')

    ax = pyplot.subplot(2, 5, 3)
    pyplot.imshow(eigFaces[:, 4].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 5')

    ax = pyplot.subplot(2, 5, 4)
    pyplot.imshow(eigFaces[:, 6].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 7')

    ax = pyplot.subplot(2, 5, 5)
    pyplot.imshow(eigFaces[:, 9].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 10')

    ax = pyplot.subplot(2, 5, 6)
    pyplot.imshow(eigFaces[:, 19].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 20')

    ax = pyplot.subplot(2, 5, 7)
    pyplot.imshow(eigFaces[:, 49].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 50')

    ax = pyplot.subplot(2, 5, 8)
    pyplot.imshow(eigFaces[:, 99].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 100')

    ax = pyplot.subplot(2, 5, 9)
    pyplot.imshow(eigFaces[:, 149].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 150')

    ax = pyplot.subplot(2, 5, 10)
    pyplot.imshow(eigFaces[:, 299].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 300')

    pyplot.subplots_adjust(top=0.85, bottom=0.01, right=0.97, left=0.03, hspace=0.1)
    pyplot.show()

def plotReconFaces(allFaces):
    pyplot.rcParams["font.family"] = 'Times New Roman'

    
    #train 
    

    #faceID = 80
    #face = allFaces.facesTrain[:,faceID]

    faceID = 111
    face = allFaces.facesTest[:,faceID]

    faceN = face - allFaces.avgTrainFace.T

    allFaces.calEigenFaces(10,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec10 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(25,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec25 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(50,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec50 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(75,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec75 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(100,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec100 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(150,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec150 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(200,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec200 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(250,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec250 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    allFaces.calEigenFaces(363,"ATA")
    w = matmul(faceN, allFaces.trEigVecs)
    rec364 = matmul(allFaces.trEigVecs, w.T) + allFaces.avgTrainFace

    fig=pyplot.figure(figsize=(5, 3.2), dpi=100)
    fig.tight_layout()
    fig.suptitle("Face reconstruction (from test dataset)", fontsize=18)
    #fig.suptitle("Face reconstruction (from train dataset)", fontsize=18)
    

    ax = pyplot.subplot(2, 5, 1)
    pyplot.imshow(face.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('Original')

    ax = pyplot.subplot(2, 5, 2)
    pyplot.imshow(rec10.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 10')

    ax = pyplot.subplot(2, 5, 3)
    pyplot.imshow(rec25.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 25')

    ax = pyplot.subplot(2, 5, 4)
    pyplot.imshow(rec50.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 50')

    ax = pyplot.subplot(2, 5, 5)
    pyplot.imshow(rec75.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 75')

    ax = pyplot.subplot(2, 5, 6)
    pyplot.imshow(rec100.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 100')

    ax = pyplot.subplot(2, 5, 7)
    pyplot.imshow(rec150.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 150')

    ax = pyplot.subplot(2, 5, 8)
    pyplot.imshow(rec200.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 200')

    ax = pyplot.subplot(2, 5, 9)
    pyplot.imshow(rec250.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 250')

    ax = pyplot.subplot(2, 5, 10)
    pyplot.imshow(rec364.reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('M = 364')

    pyplot.subplots_adjust(top=0.85, bottom=0.01, right=0.97, left=0.03, hspace=0.1)
    pyplot.show()
    exit

def compPlot(allFaces):
    # # MSE calculation
    # mseTr = zeros(363)
    # mse = zeros(363)
    # for i in range(363):
    #     allFaces.calEigenFaces(numOfEigs=i + 1, formula="ATA")
    #     mse[i] = allFaces.getMSE()[1]
    #     mseTr[i] = allFaces.getMSE()[0]
    # with open('mse.pkl', 'wb') as f:
    #     pickle.dump(mse, f)

    # Load MSE data from disk
    with open('mse.pkl', 'rb') as f:
        mse = pickle.load(f)

    # Load MSE data from disk
    with open('mseTr.pkl', 'rb') as f:
        mseTr = pickle.load(f)

    pyplot.rcParams["font.family"] = 'Times New Roman'

    #pyplot.figure(figsize=(5, 3.2), dpi=100)
    fig, ax1 = pyplot.subplots(figsize=(6, 3.5), dpi=100)
    pyplot.plot(range(1, len(mseTr)+1), mseTr)
    pyplot.plot(range(1, len(mse)+1), mse)
    
    #comp2 = log(mse) + divide(range(1, len(mse)+1), 363)
    #pyplot.plot(comp2)
    #pyplot.legend(['Train', 'Test'], fontsize=14)
    pyplot.legend(['Train', 'Test'], fontsize=14, loc=9, bbox_to_anchor=(0.745, 1.02), ncol=2)
    pyplot.xlabel("Eigenvectors used for reconstruction", fontsize=16)
    pyplot.ylabel("Mean squared error", fontsize=16)
    pyplot.title("Reconstruction MSE vs eigenfaces used", fontsize=20)
    pyplot.xticks([1] + list(range(50, 363, 50)), fontsize=12)
    pyplot.yticks(fontsize=12)
    pyplot.grid()
    pyplot.ylim(bottom=0)
    
    pyplot.text(95, 1250, "M = 87", fontsize=14)
    ax2 = ax1.twinx()
    
    complexity = log(mse) + 2*divide(range(1, len(mse)+1), 363)
    ax2.plot(complexity, color='green')
    
    ax2.set_ylabel('AIC - Model Estimator', fontsize=16, color='green')
    line = ax2.axvline(x=87, dashes=[6, 2], linewidth=1)
    line.set_color('black')
    pyplot.yticks([])
    pyplot.ylim(bottom = 5)
    
    pyplot.subplots_adjust(top=0.89, bottom=0.15, right=0.945, left=0.13, hspace=0.2)
    pyplot.show()

    
    #pyplot.show()

def getAcc(allFaces, range):
    
    trainAcc = zeros(len(range))
    testAcc = zeros(len(range))

    #with open('kNNAcc.pkl', 'rb') as f:
    #    [trainAcc, testAcc] = pickle.load(f)

    for i in range:
        if i%10 == 0:
            print(i)
        allFaces.calEigenFaces(numOfEigs=i+1, formula="ATA")
        allFaces.learnIds(nNeighbors=1)
        trAcc, teAcc = allFaces.getKNNScores()
        trainAcc[i] = trainAcc[i] + trAcc
        testAcc[i] = testAcc[i] + teAcc
    
    #with open('kNNAcc.pkl', 'wb') as f:
    #     pickle.dump([trainAcc, testAcc], f)


def errVsEV():
    with open('kNNAcc.pkl', 'rb') as f:
        [tr_kNN, te_kNN] = pickle.load(f)

    with open('RecErrAcc.pkl', 'rb') as f:
        [tr_Rec, te_Rec] = pickle.load(f)
    
    pyplot.rcParams["font.family"] = 'Times New Roman'
    ax = pyplot.plot(range(1, 364), tr_kNN)
    pyplot.plot(range(1, 364), te_kNN)
    pyplot.plot(range(52, 364, 52), tr_Rec)
    pyplot.plot(range(52, 364, 52), te_Rec)
    pyplot.title("Classification accuracies for NN and reconstruction methods", fontsize=20)
    pyplot.ylabel("Accuracy", fontsize=18)
    pyplot.xlabel("PCA bases used", fontsize=18)
    pyplot.yticks(linspace(0,1,6), ['{:,.1%}'.format(x) for x in linspace(0,1,6)], fontsize=14)
    pyplot.xticks([1] + list(range(50, 400, 50)), fontsize=14)
    pyplot.legend(['NN - Train', 'NN - Test', 'Recon. - Train', 'Recon. - Test'], fontsize=14)
    pyplot.subplots_adjust(top=0.875, bottom=0.15, right=0.955, left=0.14, hspace=0.2)
    pyplot.grid()
    pyplot.show()
    exit

#errVsEV()

def kNNSuccFail(allFaces):
    # Correct: 0, 14, 18
    #     0 (31):		137, 309, 226
    #     14 (37):	125, 148, 52
    #     18 (11):	13, 249, 171
    # Failure: 1, 3, 15, 21
    #     1 (41): 219, 138, 304
    #     3 (1): 154, 82, 282
    #     15 (32): 2, 49, 9
    #     21 (41): 9, 229, 252

    pyplot.rcParams["font.family"] = 'Times New Roman'

    
    #train 
    

    #faceID = 80
    #face = allFaces.facesTrain[:,faceID]

    faceID = 111
    face = allFaces.facesTest[:,faceID]

    faceN = face - allFaces.avgTrainFace.T

    fig=pyplot.figure(figsize=(5, 3.2), dpi=100)
    fig.tight_layout()
    fig.suptitle("Success and Failure Examples", fontsize=18)
    #fig.suptitle("Face reconstruction (from train dataset)", fontsize=18)
    

    ax = pyplot.subplot(2, 5, 1)
    pyplot.imshow(allFaces.facesTest[:, 0].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('Success')

    ax = pyplot.subplot(2, 5, 2)
    pyplot.imshow(allFaces.facesTrain[:, 137].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('neigh. 1')

    ax = pyplot.subplot(2, 5, 3)
    pyplot.imshow(allFaces.facesTrain[:, 309].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('neigh. 2')

    ax = pyplot.subplot(2, 5, 4)
    pyplot.imshow(allFaces.facesTrain[:, 226].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('neigh. 3')

    ax = pyplot.subplot(2, 5, 5)
    pyplot.imshow(allFaces.facesTrain[:, 17].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('Class sample')

    ax = pyplot.subplot(2, 5, 6)
    pyplot.imshow(allFaces.facesTest[:, 1].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('Failure')

    ax = pyplot.subplot(2, 5, 7)
    pyplot.imshow(allFaces.facesTrain[:, 219].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('neigh. 1')

    ax = pyplot.subplot(2, 5, 8)
    pyplot.imshow(allFaces.facesTrain[:, 138].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('neigh. 2')

    ax = pyplot.subplot(2, 5, 9)
    pyplot.imshow(allFaces.facesTrain[:, 304].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('neigh. 3')

    ax = pyplot.subplot(2, 5, 10)
    pyplot.imshow(allFaces.facesTrain[:, 229].reshape(46, 56).T, cmap='gray')
    pyplot.axis('off')
    ax.set_title('Class sample')

    pyplot.subplots_adjust(top=0.85, bottom=0.01, right=0.97, left=0.03, hspace=0.1)
    pyplot.show()
    exit