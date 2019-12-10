import numpy as np
from lib.functions import *


def main():
    X, Xt, y, yt, predictors, score_categories, original_y = preprocess_data(
        verbose=True, upsample=True
    )
    # plot_histogram(original_y, score_categories)
    # correlation_matrix(X.T, predictors[:-1])

    bagging(X, Xt, y, yt)
    random_forest(X, Xt, y, yt)
    boosting(X, Xt, y, yt)


if __name__ == "__main__":
    main()
