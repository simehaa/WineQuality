import numpy as np
from lib.functions import *


def main():
    X, Xt, y, yt, predictors, score_categories = preprocess_data(
        verbose=True,
        upsample=True
    )
    # covariance_matrix(X.T, predictors[:-1])
    bagging(X, Xt, y, yt)
    random_forest(X, Xt, y, yt)
    boosting(X, Xt, y, yt)


if __name__=="__main__":
    main()
