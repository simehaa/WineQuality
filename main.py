import numpy as np
from lib.functions import *


def main():
    X, Xt, y, yt, predictors, score_categories, enc = preprocess_data(
        verbose=True,
        onehot=True
    )
    random_forest(X, Xt, y, yt, predictors, score_categories, enc)


if __name__=="__main__":
    main()
