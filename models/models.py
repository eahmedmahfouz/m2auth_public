# author: ahmed mahfouz
# email: e.ahmedmahfouz@gmail.com
# Copyright 2020 The M2auth Authors. All Rights Reserved.
# =======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants_lists as cl
import paths

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def normalize_weights(weights):
    """ scaling individual samples to have unit norm
    
    Parameters:
    -----------
    weights: list of values to be normalized
    
    Returns:
    --------
    normalized_weights: normalized list of values
    """
    normalized_l = weights / np.sum(weights)
    return normalized_l


def calculate_weights(models, X, y):
    """Calculate weights for list of classification models and return the weights based on weighted majority voting algorithm
    
    Parameters
    -----------
    models: list of classification models
    X: training features set 
    y: training labels set
    
    Returns:
    --------
    sorted_w: dataframe contains of weights   
    """

    df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))
    i = 0
    for w1 in range(1, 4):
        for w2 in range(1, 4):
            for w3 in range(1, 4):

                if len(set((w1, w2, w3))) == 1:  # skip if all weights are equal
                    continue

                eclf = VotingClassifier(estimators=models, voting='soft', weights=[w1, w2, w3])

                scores = cross_val_score(estimator=eclf,
                                         X=X,
                                         y=y,
                                         cv=5,
                                         scoring='accuracy',
                                         n_jobs=1)
                df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
                i += 1

    sorted_w = df.sort_values(by=['mean', 'std'], ascending=False)
    return sorted_w.iloc[0, 0:3]


def calc_eer(fpr, tpr):
    """ calculate equal error rate
    Parameters:
    -----------
    fpr: false positive rate
    tpr: true positive rate
    
    Returns:
    --------
    eer: equal error rate
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)


def get_sample(data, sample_size=5):
    """ Return subset of a set based on sample size
    
    Parameters:
    -----------
    data: dataframe
    sample_size: maximum size of the range of the subset 
    
    Returns:
    --------
    selected_sample: dataframe as a subset of data based on sample size
    
    """
    selected_sample = pd.DataFrame()
    idL = data.groupby('id').size().index.tolist()
    for i in idL:
        selected_sample = selected_sample.append(data[data['id'] == i][:sample_size])

    return selected_sample


def get_models():
    """ Return list of learning models (algorithms)
    
    Parameters:
    ----------- 
    
    Returns:
    --------
    models: list of learning models (algorithms) 
    
    """
    models = [('LR', LogisticRegression()), ('CART', DecisionTreeClassifier()),
              ('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)),
              ('KNN', KNeighborsClassifier(n_neighbors=6, weights='distance')),
              ('RF', RandomForestClassifier(random_state=1))]
    return models


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """ plot the learning curve
    
    Parameters:
    -----------
    estimator:

    
    Returns:
    -------- 
    
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def anomaly_detection(data, clf, i, accuracy, f1score, precision, recall, eerL, modelL, userId, userprL, modelprL, fprL,
                      tprL):
    """ 
    
    Parameters:
    ----------- 
    
    Returns:
    -------- 
    
    """
    X_train_1 = data[data['user'] == 0].iloc[:, :19].values

    y_test = data['user'].values
    X_test = data.iloc[:, :19].values

    robust_scaler = StandardScaler()
    X_train_1 = robust_scaler.fit_transform(X_train_1)
    X_test = robust_scaler.transform(X_test)

    clf.fit(X_train_1)
    predictions = clf.predict(X_test)
    predictions = [0 if x == 0.0 else x for x in predictions]
    predictions = [1 if x == 'anomaly' else x for x in predictions]

    modelL.append('LSAnomaly')
    userId.append(i)
    accuracy.append(accuracy_score(y_test, predictions))
    f1score.append(f1_score(y_test, predictions, average='weighted'))
    precision.append(precision_score(y_test, predictions, average='weighted'))
    recall.append(recall_score(y_test, predictions, average='weighted'))
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    print(accuracy_score(y_test, predictions), f1_score(y_test, predictions, average='weighted'))
    print('eer', calc_eer(fpr, tpr))
    print('----------')
    eerL.append(calc_eer(fpr, tpr))
    [fprL.append(x) for x in fpr.tolist()]
    [tprL.append(x) for x in tpr.tolist()]
    [modelprL.append(x) for x in (['lsanomaly'] * len(fpr))]
    [userprL.append(x) for x in ([i] * len(fpr))]


def one_versus_all():
    userprL = []
    probabilityCallL = []
    predictionCallL = []
    probabilityProximityL = []
    predictionProximityL = []
    probabilityLocationL = []
    predictionLocationL = []
    probabilityWlanL = []
    predictionWlanL = []
    probabilityKeystrokeL = []
    predictionKeystrokeL = []
    probabilityGestureL = []
    predictionGestureL = []
    modelL = []
    actualL = []

    # load data of all modalities
    data = pd.read_csv(paths.mit_reality_dataset + 'all_modalities.csv', header=None)
    data.columns = cl.all_modalities
    original_data = data.copy()

    # loop over all other users as imposter and calculate the model accuracy
    for i in data.groupby('user').size().index.tolist():
        data = original_data.copy()
        sample = get_sample(data[data['user'] != i])
        data = sample.append(data[data['user'] == i])
        data['user'] = np.where(data['user'] == i, 0, 1)
        print(data.groupby('user').size())

        # avoid tiny samples
        if len(data[data['user'] == 0]) < 100:
            continue

        y = data['user'].values
        X = data.iloc[:, 1:50].values

        # Split-out validation dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        actualL = actualL + y_test.tolist()
        userprL = userprL + ([i] * len(y_test))
        modelL = modelL + (['eclf'] * len(y_test))
        modelnameFlag = 0
        for X_train, X_test, modality in [[X_train[:, 0:2], X_test[:, 0:2], 'call'],
                                          [X_train[:, 2:4], X_test[:, 2:4], 'proximity'],
                                          [X_train[:, 4:6], X_test[:, 4:6], 'location'],
                                          [X_train[:, 6:8], X_test[:, 6:8], 'wlan'],
                                          [X_train[:, 8:27], X_test[:, 8:27], 'keystroke'],
                                          [X_train[:, 27:49], X_test[:, 27:49], 'gesture']]:
            print(modality)
            # scale data using standard scalar
            robust_scaler = StandardScaler()
            X_train = robust_scaler.fit_transform(X_train)
            X_test = robust_scaler.transform(X_test)
            models = get_models()
            eclf = VotingClassifier(estimators=models, voting='soft')
            eclf.fit(X_train, y_train)
            prediction = eclf.predict(X_test)

            # add prediction data
            if modality == 'call':
                predictionCallL = predictionCallL + prediction.tolist()
            elif modality == 'proximity':
                predictionProximityL = predictionProximityL + prediction.tolist()
            elif modality == 'location':
                predictionLocationL = predictionLocationL + prediction.tolist()
            elif modality == 'wlan':
                predictionWlanL = predictionWlanL + prediction.tolist()
            elif modality == 'keystroke':
                predictionKeystrokeL = predictionKeystrokeL + prediction.tolist()
            else:
                predictionGestureL = predictionGestureL + prediction.tolist()

            # add probability data
            if modality == 'call':
                probabilityCallL = probabilityCallL + eclf.predict_proba(X_test)[:, 1].tolist()
            elif modality == 'proximity':
                probabilityProximityL = probabilityProximityL + eclf.predict_proba(X_test)[:, 1].tolist()
            elif modality == 'location':
                probabilityLocationL = probabilityLocationL + eclf.predict_proba(X_test)[:, 1].tolist()
            elif modality == 'wlan':
                probabilityWlanL = probabilityWlanL + eclf.predict_proba(X_test)[:, 1].tolist()
            elif modality == 'keystroke':
                probabilityKeystrokeL = probabilityKeystrokeL + eclf.predict_proba(X_test)[:, 1].tolist()
            else:
                probabilityGestureL = probabilityGestureL + eclf.predict_proba(X_test)[:, 1].tolist()

            for name, model in models:
                print(name)
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                if modelnameFlag == 0:
                    userprL = userprL + ([i] * len(y_test))
                    modelL = modelL + ([name] * len(y_test))
                    actualL = actualL + y_test.tolist()

                # add prediction data
                if modality == 'call':
                    predictionCallL = predictionCallL + prediction.tolist()
                elif modality == 'proximity':
                    predictionProximityL = predictionProximityL + prediction.tolist()
                elif modality == 'location':
                    predictionLocationL = predictionLocationL + prediction.tolist()
                elif modality == 'wlan':
                    predictionWlanL = predictionWlanL + prediction.tolist()
                elif modality == 'keystroke':
                    predictionKeystrokeL = predictionKeystrokeL + prediction.tolist()
                else:
                    predictionGestureL = predictionGestureL + prediction.tolist()

                # add probability data
                if modality == 'call':
                    probabilityCallL = probabilityCallL + model.predict_proba(X_test)[:, 1].tolist()
                elif modality == 'proximity':
                    probabilityProximityL = probabilityProximityL + model.predict_proba(X_test)[:, 1].tolist()
                elif modality == 'location':
                    probabilityLocationL = probabilityLocationL + model.predict_proba(X_test)[:, 1].tolist()
                elif modality == 'wlan':
                    probabilityWlanL = probabilityWlanL + eclf.predict_proba(X_test)[:, 1].tolist()
                elif modality == 'keystroke':
                    probabilityKeystrokeL = probabilityKeystrokeL + eclf.predict_proba(X_test)[:, 1].tolist()
                else:
                    probabilityGestureL = probabilityGestureL + eclf.predict_proba(X_test)[:, 1].tolist()

            modelnameFlag = 1

    performance = pd.DataFrame(
        [userprL, modelL, probabilityCallL, predictionCallL, probabilityProximityL, predictionProximityL,
         probabilityLocationL, predictionLocationL, probabilityWlanL, predictionWlanL,
         probabilityKeystrokeL, predictionKeystrokeL, probabilityGestureL, predictionGestureL, actualL])
    performance = performance.transpose()
    performance.columns = cl.performance_profile_all
    print(performance)


if __name__ == '__main__':
    one_versus_all()
