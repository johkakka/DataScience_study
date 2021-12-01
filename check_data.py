import pandas as pd

df = pd.read_csv('data/hotel_bookings.csv')


df_fill = df.drop('company', axis=1)
df_fill =df_fill.fillna({'children':0, 'country':'NNN', 'agent':-1})
#print(df.isnull().all())
print(df_fill)




