# 2019115177 배주한
import pandas as pd

data = {
'year':[2016, 2017, 2018],
'car': ['그랜저', '그랜저', '소나타'],
'name': ['홍길동', '고길동', '김둘리' ],
'number' : ['123하4567', '123허4567', '123호4567']
}

#%%
data_df = pd.DataFrame(data)
print(data_df)

#%%
data_df.loc[4] = [2017, '일론', '테슬라', '987하6543']
print(data_df)

#%%
print(data_df[['year', 'car', 'number']])

#%%
print(data_df[ data_df['year']<2018])