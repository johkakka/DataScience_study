import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pd.read_csv('hotel_bookings.csv')
    # data.columns = ["hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    #                 "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    #                 "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    #                 "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
    #                 "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type",
    #                 "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list", "customer_type",
    #                 "adr", "required_car_parking_spaces", "total_of_special_requests", "reservation_status",
    #                 "reservation_status_date"]
    df_fill = data.fillna({'children': 0, 'country': 'NNN', 'agent': -1})
    df_fill = df_fill.reindex(columns=["is_canceled","hotel", "lead_time", "arrival_date_year", "arrival_date_month",
                                       "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
                                       "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
                                       "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
                                       "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type",
                                       "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list", "customer_type",
                                       "adr", "required_car_parking_spaces", "total_of_special_requests", "reservation_status",
                                       "reservation_status_date"])
    df_fill = df_fill.drop('company', axis=1)

    # Dummy variable
    # Nominal scale
    data_le = pd.get_dummies(df_fill, drop_first=True,
                             columns=["hotel", "is_canceled", "meal", "country","market_segment", "distribution_channel", "is_repeated_guest", "reserved_room_type", "assigned_room_type", "deposit_type", "agent", "customer_type", "reservation_status"])
    # Ordinal scale
    for column in ["lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests", "reservation_status_date"]:
        selected_column = data_le[column]
        le = preprocessing.LabelEncoder()
        le.fit(selected_column)
        column_le = le.transform(selected_column)
        data_le[column] = pd.Series(column_le).astype('category')

    # split training data & test data
    # X: other (Explanatory variable)
    # Y: class
    (X_train, X_test, y_train, y_test) = train_test_split(data_le.iloc[:, 1:], data_le.iloc[:, 0],
                                                          test_size=0.1, random_state=0)

    # Run
    forest = RandomForestRegressor(n_estimators=20, criterion='mse', random_state=1, n_jobs=-1)
    # forest = RandomForestClassifier(n_estimators=10, random_state=1, n_jobs=-1)
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
                     color="orange")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="orange",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'x-', color="b",
             label="Cross-validation score")
    # plt.xlabel('sample size')
    # plt.ylabel('score')
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
    indices = indices[:20]

    for i in range(len(indices)):
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
