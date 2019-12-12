import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from lib.functions import *


def main():
    X, Xt, y, yt, predictors, score_categories, original_y = preprocess_data(
        verbose=True, upsample=False
    )
    # plot_histogram(original_y, score_categories)
    # correlation_matrix(X, y, predictors)

    bagging(X, Xt, y, yt)
    random_forest(X, Xt, y, yt)
    # boosting(X, Xt, y, yt)

    # plot_feature_importances(
        # X, y, RandomForestClassifier(n_estimators=900, max_features=11), predictors
    # )
    # plot_feature_importances(
        # X, y, GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1), predictors
    # )

if __name__ == "__main__":
    main()
