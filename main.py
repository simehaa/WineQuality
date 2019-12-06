import numpy as np
from lib.functions import *


def main():
    X, Xt, y, yt, predictors, score_categories = preprocess_data(verbose=True)
    random_forest(X, Xt, y, yt)


if __name__=="__main__":
    main()
