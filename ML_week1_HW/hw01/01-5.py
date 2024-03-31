# 2019115177 배주한
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

csv_2014 = pd.read_csv('C:/Users/spbab/Desktop/4-1/ML_HW/ML_week1_HW/2014.csv', encoding='CP949')
csv_2015 = pd.read_csv('C:/Users/spbab/Desktop/4-1/ML_HW/ML_week1_HW/2015.csv', encoding='CP949')
csv_2016 = pd.read_csv('C:/Users/spbab/Desktop/4-1/ML_HW/ML_week1_HW/2016.csv', encoding='CP949')

#%%
# (1) 불러온 3개 데이터 병합, '구분'을 index
df = pd.concat([csv_2014, csv_2015, csv_2016])
df = df.set_index(['구분'])
print(df)
print()
#%%
# (2) 년도, 월별로 index를 설정(멀티인덱스)하고 데이터를 보이기
df.index = df.index.str.split('년|월').map(lambda x: (int(x[0]), int(x[1])))
df.index.names = ['년도', '월']
print(df)
print()
#%%
# (3) 년도 및 월별로 사망자 출력.
mean_year = df.groupby('년도').mean()
print(mean_year.iloc[:,[1]])
mean_month = df.groupby('월').mean()
print(mean_month.iloc[:,[1]])
print()
#%% (4) 2016년 전체교통사고대비 사망률을 출력
df_2016 = df.loc[2016].sum(axis=0)
print('2016년 통계\n전체사고(건):{}, 사망자(명):{}, 사고대비사망율:{:.2f}%'.format(df_2016.iloc[0],
                                        df_2016.iloc[1], df_2016.iloc[1]/df_2016.iloc[0]*100))
print()
#%% (5) 2014년도 월별 사망, 부상 데이터를 bar차트로 표현
df_2014 = df.loc[2014, ['사망(명)', '부상(명)']]
df_2014.plot(kind='bar')
print()
#%% (6) 2015년 대비 사망이 가장 많이 증가한 2016년도 2개의 월을 구하시오.
x = df.loc[2016,['사망(명)']]-df.loc[2015,['사망(명)']]
x_sorted = x.sort_values(['사망(명)'],ascending=False).head(2)
print(x_sorted)
print()