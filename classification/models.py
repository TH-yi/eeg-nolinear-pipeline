from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np


def simple_svm(X, y, test_size=0.2, random_state=42):
    """Return accuracy of an RBF‑kernel SVM on the given feature matrix."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale"))
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)
