import matplotlib.pyplot as plt
import numpy as np
#import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig=plt.figure(figsize=(5.5, 5), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    tick_marks = range(9, max(classes), 10)
    plt.xticks(tick_marks, range(10, max(classes)+1, 10), fontsize=14)
    plt.yticks(tick_marks, range(10, max(classes)+1, 10), fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plt.text(j, i, format(cm[i, j], fmt),
        #         horizontalalignment="center",
        #         color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.subplots_adjust(top=0.899, bottom=0.133, right=1, left=0.14)
