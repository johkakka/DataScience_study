import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def main():
    data = pd.read_csv('mushrooms.csv')
    data.columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                    "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
                    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
                    "spore-print-color", "population", "habitat"]

    # Dummy variable
    # Nominal scale
    data_le = pd.get_dummies(data, drop_first=True,
                             columns=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                                      "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                                      "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
                                      "veil-type", "veil-color", "ring-type", "spore-print-color", "habitat"])
    # Ordinal scale
    for column in ["class", "gill-spacing", "gill-size", "ring-number", "population"]:
        selected_column = data_le[column]
        le = preprocessing.LabelEncoder()
        le.fit(selected_column)
        column_le = le.transform(selected_column)
        data_le[column] = pd.Series(column_le).astype('category')

    # split training data & test data
    # X: other (Explanatory variable)
    # Y: class
    (X_train, X_test, y_train, y_test) = train_test_split(data_le.iloc[:, 1:], data_le.iloc[:, 0],
                                                          test_size=0.3, random_state=0)
    X_train.to_csv('data/x_train.csv')
    X_test.to_csv('data/x_test.csv')
    y_train.to_csv('data/y_train.csv')
    y_test.to_csv('data/y_test.csv')

    # Run
    forest = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    # plot
    train_sizes=np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
        forest, X_train, y_train, cv=3, train_sizes=train_sizes, random_state=42, shuffle=True
    )

    print("train_sizes(検証したサンプル数): {}".format(train_sizes))
    print("------------")
    print("train_scores(各サンプル数でのトレーニングスコア): \n{}".format(train_scores))
    print("------------")
    print("test_scores(各サンプル数でのバリデーションスコア): \n{}".format(test_scores))


if __name__ == '__main__':
    main()
