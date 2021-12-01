import pandas as pd

df = pd.read_csv('hotel_bookings.csv')

#print(df.isnull().sum())
#null = children4, country488, agent16340, company112593
#children null=0, country null = NNN   agent null = -1, company なくす
df_fill = df.drop('company', axis=1)

df_fill = pd.Series([0, 'NNN', -1], index=['children', 'country','agent'])
print(df_fill.isnull().sum())





