import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, \
    RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
np.random.seed(7)


def preprocess_data(verbose=False, onehot=False):
    """
    Read the data file of red wine quality, a data set with 1599 data points,
    11 features and one categorical outcome (wine quality scored 0-10). The
    data set can be viewed as both a classification problem and a regression
    problem. Data set info:

    Predictors
    ----------
    fixed acidity
    volatile acidity
    citric acid
    residual sugar
    chlorides
    free sulfur dioxide
    total sulfur dioxide
    density
    pH
    sulphates
    alcohol

    Response
    --------
    quality : score 0-10

    Parameters
    ----------
    verbose : boolean
        Option to print a data set summary after preprocessing the data

    Returns
    -------
    X, Xt : arrays, shape(N, p)
        Train and test data respectively,
        with shapes N data posints with p predictors

    y, yt : arrays, shape(N, 11)
        One-hot encoded response data with scores from 0-10 (11 categories)

    predictors : list
        List of all 12 feature names (11 predictors + 1 response)
    """
    folder = Path("data/")
    df = pd.read_csv(folder / "winequality-red.csv", sep=";")
    X = df.loc[:, df.columns != "quality"].values
    y = df["quality"].values
    predictors = list(df) # list of all columns including response
    # Split of train and test set
    X, Xt, y, yt = train_test_split(X, y, test_size=0.33)
    # Scale both the train/test according to training set:
    scl = StandardScaler()
    scl.fit(X)
    X = scl.transform(X)
    Xt = scl.transform(Xt)
    # Upscale so that the number of outputs in each class are equal
    upscale = RandomOverSampler()
    X, y = upscale.fit_resample(X, y.astype("int"))
    # One hot encode the categorical outcome data
    # NOTE: It turns out that the data set only contains the categories
    # [3, 4, 5, 6, 7, 8] which are only 6 categories, instead of 11.
    enc = OneHotEncoder(sparse=False)
    y_enc = enc.fit_transform(y.reshape(-1, 1))
    yt_enc = enc.fit_transform(yt.reshape(-1, 1))
    score_categories = enc.categories_[0]
    num_categories = len(score_categories)
    # print a pretty overview of the data
    if verbose:
        # Info about X and Xt
        print("\n\t------------- TABLE 1: processed X data summary. ----------")
        print("\tPredictor   | train mean | train std | test mean | test std")
        print("\t------------|------------|-----------|-----------|---------")
        for i,name in enumerate(predictors[:-1]): # exclude response (from list of names)
            if len(name) > 11:
                name = name[:9] + ".."
            print(f"\t{name:11.11} ", end="")
            print(f"| {np.mean(X[:,i]):10.2e} | {np.std(X[:,i]):9.2f} ", end="")
            print(f"| {np.mean(Xt[:,i]):9.2e} | {np.std(Xt[:,i]):8.2f}")
        # Info about y and yt
        print("\n\t-- TABLE 2: processed y data summary ---")
        print("\tScore | train occurence | test occurence")
        print("\t------|-----------------|---------------")
        for i in range(num_categories):
            score = score_categories[i]
            ysum = np.sum(y_enc[:,i])
            ytsum = np.sum(yt_enc[:,i])
            print(f"\t{score:}     | {int(ysum):15} | {int(ytsum):14}")
        print("\n")

    if onehot:
        return X, Xt, y_enc, yt_enc, predictors, score_categories, enc
    else:
        return X, Xt, y, yt, predictors, score_categories, enc


def random_forest(X, Xt, y, yt, predictors, score_categories, enc):
    # Fit/predict
    clf = RandomForestClassifier(n_estimators=500, max_features="sqrt")
    clf.fit(X, y)
    yp = clf.predict(Xt)
    yt = enc.inverse_transform(yt)
    yp = enc.inverse_transform(yp)
    C = confusion_matrix(yt, yp, normalize="true")
    plt.show()
    # Plot
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        C,
        ax=ax,
        cmap="Blues",
        annot=True,
        xticklabels=score_categories,
        yticklabels=score_categories
    )
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
