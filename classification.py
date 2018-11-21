from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE

def cross_validation_svm(data, label, kernel="rbf", gamma="scale", cv=5):
    clf = svm.SVC(kernel=kernel, gamma=gamma)
    scores = cross_val_score(clf, data, label, cv=cv)
    print(f"Accuracy: {scores.mean()}(+/- {scores.std() * 2})")

def dimension_reduction(data, method="TLSE", ndim=2):
