from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_svm(data, label, plot=False, test_rate=0.33):
    if plot:
        data = _dimension_reduction(data)
    X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                        test_size=test_rate)
    clf = svm.SVC(kernel="rbf", gamma="scale")
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    plot_svm_boundaries(clf, data, label)

    _plot_confusion_matrix(y_test, y_pred, np.unique(label.values))
    return clf, X_test, y_test, y_pred


def predict_svm(classifier, data_windowed, nb_points, T):
    predicted_label_windowwise = classifier.predict(data_windowed)
    # the point j belongs to the T windows: j-(T-1), ..., j
    # majority vote: the label of a point is 1 if ceil(T/2) windows votes 1
    # that means the sum/T >= 0.5, ie int(2*sum/T) = 1
    predicted_label_pointwise = []
    for k in range(T, len(data)):
        su = np.sum(predicted_label_windowwise[k-T])
        prediction_one_label = int(2*su/T)
        predicted_label_pointwise.append(prediction_one_label)
    return predicted_label_pointwise



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


def _plot_confusion_matrix(y_test, y_pred, class_names,
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
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    _plot_svc_decision_function(clf)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
