import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pd.read_csv('mushrooms.csv')
    data.columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                    "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
                    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
                    "spore-print-color", "population", "habitat"]

    # drop column
    data = data.drop(columns=["cap-shape", "cap-surface", "cap-color", "gill-attachment",
                    "gill-spacing", "gill-color", "stalk-shape", "stalk-root",
                    "stalk-color-above-ring",
                    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
                    "population", "habitat"])

    # Dummy variable
    # Nominal scale
    # data_le = pd.get_dummies(data, drop_first=True,
                             # columns=["cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                             #          "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                             #          "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring",
                             #          "veil-type", "veil-color", "ring-type", "spore-print-color", "habitat"])
    data_le = pd.get_dummies(data, drop_first=True,
                             columns=["odor", "bruises", "stalk-surface-above-ring",
                                      "stalk-surface-below-ring", "ring-type", "spore-print-color"])
    # Ordinal scale
    for column in ["class", "gill-size"]:
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
    # forest = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)
    forest = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    # plot
    train_sizes=np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
        forest, X_train, y_train, cv=5, train_sizes=train_sizes, random_state=42, shuffle=True
    )

    print("train_sizes(検証したサンプル数): {}".format(train_sizes))
    print("------------")
    print("train_scores(各サンプル数でのトレーニングスコア): \n{}".format(train_scores))
    print("------------")
    print("test_scores(各サンプル数でのバリデーションスコア): \n{}".format(test_scores))

    plt.figure()
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="m")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="m",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'x-', color="g",
             label="Cross-validation score")
    plt.xlabel('sample size')
    plt.ylabel('score')
    plt.title('Learning Curves')
    plt.legend(loc="best")

    # テストデータでの正解率を出力
    accuracy = forest.score(X_test, y_test)
    print("test data accuracy:", accuracy)

    plt.show()


    # 変数ごとの重要度をみる

    feature = forest.feature_importances_
    # 特徴量の重要度を上から順に出力する
    f = pd.DataFrame({'number': range(0, len(feature)),
                      'feature': feature[:]})
    f2 = f.sort_values('feature', ascending=False)

    # 特徴量の名前
    label = data_le.columns[0:][1:]

    # 特徴量の重要度順（降順）
    indices = np.argsort(feature)[::-1]

    for i in range(len(feature)):
        print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

    plt.figure()
    plt.title('Feature Importance')
    plt.bar(range(len(feature)), feature[indices], color='lightblue', align='center')
    plt.xticks(range(len(feature)), label[indices], rotation=90)
    plt.xlim([-1, len(feature)])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
