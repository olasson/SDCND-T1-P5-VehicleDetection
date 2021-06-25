import numpy as np
import cv2

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(features_vehicles, features_non_vehicles, random_state = None):

    # Data
    X = np.vstack((features_vehicles, features_non_vehicles)).astype(np.float64)

    # Labels
    y = np.hstack((np.ones(len(features_vehicles)), np.zeros(len(features_non_vehicles))))

    if random_state is None:
        random_state = np.random.randint(0, 100)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = random_state)

    X_scaler = StandardScaler().fit(X_train)

    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, X_scaler


def svc_train(X, y):

    svc = LinearSVC()

    svc.fit(X, y)

    return svc

def svc_accuracy(svc, X, y):

    accuracy = round(svc.score(X, y), 3)

    return accuracy
