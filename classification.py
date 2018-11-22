from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_svm(data, label, reduction=False, test_rate=0.33):
    if reduction:
        data = _dimension_reduction(data, method=reduction)
    X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                        test_size=test_rate)
    clf = svm.SVC(kernel="rbf", gamma="scale")
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    _print_confusion_matrix(y_test, y_pred, np.unique(label.values))
    return clf, X_test, y_test, y_pred


def point_belongs_to_windows(index, size_window, step):
    n_w = int(size_window/  step)
    last_window = int((index - size_window)/step)
    first_window = max(0, last_window - n_w+1)
    real_last_window = max(first_window, last_window)
    windows = range(first_window, real_last_window)
    return windows


def predict_svm(classifier, data_windowed, nb_points, size_window, step):
    predicted_label_windowwise = classifier.predict(data_windowed)
    # the point j belongs to the  windows: j-(T-1), ..., j
    # majority vote: the label of a point is 1 if ceil(T/2) windows votes 1
    # that means the sum/T >= 0.5, ie int(2*sum/T) = 1
    predicted_label_pointwise = [0 for k in range(size_window)]
    # n_w: number of windows a point belongs to
    for k in range(size_window, nb_points):
        windows = point_belongs_to_windows(k, size_window, step)
        su = np.sum([predicted_label_pointwise[w] for w in windows])
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


def _print_confusion_matrix(y_test, y_pred, class_names,
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
