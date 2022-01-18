import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier


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
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    # y_train = preprocessing.scale(y_train)
    # y_test = preprocessing.scale(y_test)

    # Run
    f = MLPClassifier(solver="sgd", random_state=0, activation='tanh', max_iter=10000)

    f.fit(X_train, y_train)

    accuracy_train = f.score(X_train, y_train)
    print("accuracy (Train):", accuracy_train)
    accuracy = f.score(X_test, y_test)
    print("accuracy (Test):", accuracy)
    plt.title("Loss Curve")
    plt.plot(f.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    # テストデータでの正解率を出力
    accuracy = f.score(X_test, y_test)
    print("test data accuracy:", accuracy)

    plt.show()

if __name__ == '__main__':
    main()
