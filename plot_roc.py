from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(clf_fited, data_windowed, real_labels):
    '''
    data: data whose labels are known
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                        test_size=test_rate)
    nb_points = len(real_labels)
    y_score = predict_svm(clf_fited, data_windowed, n, nb_points, TIMEWINDOW, STEP)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
