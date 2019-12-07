import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from imblearn.over_sampling import RandomOverSampler
# Scikit-learn:
from sklearn.model_selection import \
    train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import \
    GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Global parameters
np.random.seed(7)
DATA_FOLDER = Path("data/")
FIG_FOLDER = Path("fig/")


def preprocess_data(verbose=True, upsample=True):
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

    upsample : boolean
        Option to upsample training data to have equal amounts of Data
        for each category.

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
    df = pd.read_csv(DATA_FOLDER / "winequality-red.csv", sep=";")
    X = df.loc[:, df.columns != "quality"].values
    y = df["quality"].values
    predictors = list(df) # list of all columns including response
    for i,name in enumerate(predictors):
        if len(name) > 11:
            name = name[:9] + ".."
            predictors[i] = name
    # Split of train and test set
    X, Xt, y, yt = train_test_split(X, y, test_size=0.33)
    # Scale both the train/test according to training set:
    scl = StandardScaler()
    scl.fit(X)
    X = scl.transform(X)
    Xt = scl.transform(Xt)
    # Upscale so that the number of outputs in each class are equal
    if upsample:
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
            print(f"\t{score:1.1f}   | {int(ysum):15} | {int(ytsum):14}")
        print("\n")
    return X, Xt, y, yt, predictors, score_categories


def covariance_matrix(X, predictors):
    sigma = np.cov(X)
    ax = plt.subplot(111)
    ax.set_title(f"Covariance Matrix")
    sns.heatmap(
        sigma,
        ax=ax,
        vmin=-1.0,
        vmax=1.0,
        cmap="Blues",
        xticklabels=predictors,
        yticklabels=predictors
    )
    b, t = plt.ylim() # discover the values for bottom and top
    plt.ylim(b+0.5, t-0.5)
    plt.savefig(FIG_FOLDER / "cov_matrix.png")
    plt.close()
    return None


def accuracy(clf, X, y, cv=5):
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    acc = np.mean(scores)
    return acc


def plot_confusion_matrix(X, Xt, y, yt, clf):
    clf.fit(X, y)
    yp = clf.predict(Xt)
    C = confusion_matrix(yt, yp, normalize='true')
    # Method and CV accuracy
    method_name = type(clf).__name__
    acc = accuracy(clf, X, y)
    # Plot
    ax = plt.subplot(111)
    ax.set_title(method_name + f", accuracy = {acc:1.2f}")
    sns.heatmap(
        C,
        ax=ax,
        cmap="Blues",
        annot=True,
        xticklabels=range(3,9),
        yticklabels=range(3,9),
        square=True
    )
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.show()


def bagging(X, Xt, y, yt):
    clf = BaggingClassifier(
        n_estimators=500,
        bootstrap=True,
        n_jobs=-1,
        oob_score=True
    )
    plot_confusion_matrix(X, Xt, y, yt, clf)


def random_forest(X, Xt, y, yt):
    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=-1,
    )
    plot_confusion_matrix(X, Xt, y, yt, clf)


def boosting(X, Xt, y, yt):
    clf = GradientBoostingClassifier(
        n_estimators=100
    )
    plot_confusion_matrix(X, Xt, y, yt, clf)
