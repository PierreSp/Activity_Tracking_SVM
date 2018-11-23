from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def predict_svm(classifier, data_windowed, nb_points, size_window, step):
    predicted_label_windowwise = classifier.predict(data_windowed)
    # the point j belongs to the  windows: j-(T-1), ..., j
    # majority vote: the label of a point is 1 if ceil(T/2) windows votes 1
    # that means the sum/T >= 0.5, ie int(2*sum/T) = 1
    predicted_label_pointwise = [0 for k in range(size_window)]
    # n_w: number of windows a point belongs to
    for k in range(size_window, nb_points):
        windows = point_belongs_to_windows(k, size_window, step)
        try:
            su = np.sum([predicted_label_windowwise[w] for w in windows])
            prediction_one_label = int(min(1, int(17*su/len(windows))))
        except Exception as ex:
            prediction_one_label = 0
        predicted_label_pointwise.append(prediction_one_label)
    return predicted_label_pointwise, predicted_label_windowwise


def point_belongs_to_windows(index, size_window, step):
    n_w = int(size_window /  step)
    last_window = int((index - size_window)/step)
    first_window = max(0, last_window - n_w + 1)
    real_last_window = max(first_window, last_window)
    windows = range(first_window, real_last_window)
    return windows


def cross_validation_svm(data, label, kernel="rbf", gamma="scale", cv=5):
    clf = svm.SVC(kernel=kernel, gamma=gamma)
    scores = cross_val_score(clf, data, label, cv=cv)
    print(f"Accuracy: {scores.mean()}(+/- {scores.std() * 2})")


def _dimension_reduction(data, method="TSNE", ndim=2):
    if method == "TSNE":
        embedded_data = TSNE(n_components=ndim).fit_transform(data)
    elif method == "PCA":
        embedded_data = PCA(n_components=ndim).fit_transform(data)
    else:
        print(f"Method {method} not available. Use TLSE instead")
        embedded_data = TSNE(n_components=ndim).fit_transform(data)
    return embedded_data


def plot_roc_curve(clf, X_test, y_test):
    y_score = clf.decision_function(X_test)
    y_test =  label_binarize(y_test, classes=[0, 1])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def plot_confusion_matrix(y_test, y_pred, class_names,
                           normalize=False, figsize=(10, 7), fontsize=30):
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    df_cm = pd.DataFrame(
        cm, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        sns.heatmap(df_cm, square=True, annot=True, fmt='.2f' if normalize
                    else 'd', cbar=False)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print(fig)


def _plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_svm_boundaries(clf, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='Dark2', label=X[:, 1])
    _plot_svc_decision_function(clf)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')


def plot_roc_train(clf_fited, data, label):
    '''
    data: data whose labels are known
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                        test_size=0.33)
    n = len(label)
    T = len(label)-len(data)
    y_score = clf_fited.predict(X_test)
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
