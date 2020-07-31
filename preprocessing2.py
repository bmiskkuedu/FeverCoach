#%%

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import io
import csv
import glob
from tqdm import tqdm_notebook as tqdm
from ipywidgets import IntProgress
from IPython.display import display
import collections
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

#%%

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print('x' in np.arange(5))

#%%

path_in = r''
path_out = r'\{}'
path_save = r''
class_translate = {"flu_true" : 1, "cold" : 0}
pd.options.mode.chained_assignment = None  # default='warn'


#%%

#raise
size_checker = []
discarded_file = []
weekNumList = []
for k in class_translate.keys():
    print("Processing set " + k)
    all_files = glob.glob(os.path.join(path_in, k, "*.csv"))
    for file in tqdm(all_files):
        try:
            file_name = os.path.splitext(os.path.basename(file))[0]
            df = pd.read_csv(file)
            df.index.name = file_name
            temp1 = df.columns.tolist()
            if 'True' in temp1:
                df=df.drop(df.columns[0], axis=1)
            # Drop a row by condition
            df = df.loc[(df.is_test != 1) & (df.is_delete != 1)]
            #temp = df.loc[(df['type']==2) & (df['data_1']=='1')] ###antibiotics != antivirals
            #print(temp)
            df2 = df.loc[(df['type']==1) | ((df['type']==2) & (df['data_1']=='0'))]
            df3 = df.loc[(df['type']==4)]
            df4 = df.loc[(df['type']==2) & (df['data_1']=='1')]
            ######type 1 and 2
            list01 = ['date', 'type', 'data_1', 'data_2', 'data_3', 'data_4'] #data3 제형 data4 성분
            df21 = df2[df2.columns.intersection(list01)].copy()
            df21['data_2'].loc[df21['data_2'] == 'C'] = float(0) ##may userwarning
            df21 = df21.astype({"data_1": float, "data_2": float})
            #print(df21)
            df21['value'] = df21['data_1'].copy() + df21['data_2'].copy()
            df22 = df21[['date', 'type', 'value']].copy()
            df22['type'] = df21['type'].copy().map({1: 'fever', 2: 'reducer'})
            #print(df22)
            list02=['date', 'type', 'data_1', 'data_2', 'data_3', 'data_4', 'memo'] #data3 제형 data4 성분
            df31 = df3[df3.columns.intersection(list02)].copy()
            #print(df31) ##1 : 증상   2 : 항생제   3 : 병원진단   4 : 오늘 일   5 : 예방접종
            df32 = df31.loc[(df31['data_1']=='1')] ##증상
            df33 = df32[['date', 'type', 'data_2']].copy()
            #df33 = df32[['date', 'type', 'memo']].copy()
            df34 = df33.rename(columns = {'data_2':'value'})
            temp31 = df34['value'].unique()
            temp32 = []
            for item in temp31:
                temp32.append(item.split('_'))
            #temp33 = temp32.squeeze()
            temp33 = (list(set([item for sublist in temp32 for item in sublist])))
            temp34 = [int(x) for x in temp33]
            temp35 = sorted(temp34)
            #print(df34)
            #print(temp33)
            #df34["new_column"] = temp33
            df35 = pd.DataFrame(columns=['date', 'type', 'value'])
            for i in range(len(temp35)):
                df35.loc[i] =  [min(df34['date']), 'symptoms', int(temp35[i])]
            '''
             if len(df34) == len(temp33):
                pass
            else:
                for item in temp33:
                    df34['value']          
            '''
            #print(type(temp3))
            #next(iter(all_files))
            df41 = df4[df4.columns.intersection(list01)].copy()
            df41['data_2'].fillna(0, inplace=True) ## 2-0-x은 해열제, 2-1-0은 감기약, 2-2-???는 항생제
            df42 = df41[['date', 'type', 'data_1']].copy()
            df43 = df42.replace({'type': 2}, 9) ###9==antibiotics
            df44 = df43.rename(columns = {'data_1':'value'})
            #print(df44)
            #####weeknumber
            #df2=df2.drop(df.columns[0], axis=1)
            #week_number_counter = (min(df2['date'])[0:4], min(df2['date'])[5:7], min(df2['date'])[8:10])
            #datetime.datetime.strptime('24052010', "%d%m%Y").date()
            #print(datetime.date(convert_date).isocalendar()[1])
            #datetime.datetime.strptime
            #datetime.date(2010, 6, 16).isocalendar()[1] datetime.date(2010, 6, 16).strftime("%V")
        except:
            discarded_file.append([file_name, 'preprocessing_error'])
        try:
            if k == 'flu_true' :
                df22.loc['-2'] = [min(df22['date']), 'isFlu', int(1)]
            else:
                df22.loc['-2'] = [min(df22['date']), 'isFlu', int(0)]
            #print(1)
            df22.loc['-1'] = [min(df22['date']), 'babyid', int(df['baby_id'].unique())]
            #print(df22)
            #df2 = df2.drop(df2[(df2.type == 2) & (df.data_1 == 1)].index)
            #index = df22['value'].index[df22['value'].apply(np.isnan)].copy()
            #for i in df22.index:
            #    if i in index:
            #        df22.loc[i, 'type']='cold medicine'
            #        df22.loc[i, 'value']=1
            convert_date = datetime.strptime(min(df22['date']), '%Y-%m-%d %H:%M:%S')
            week_number = datetime.date(convert_date).strftime("%V") ###date.isocalendar ()
            #print(week_number)
            df22.loc['-3'] = [min(df22['date']), 'weekNumber', week_number]
            if k == 'flu_true':
                weekNumList.append(week_number)
            if df35.empty:
                #pass
                discarded_file.append([file_name, 'no_symptom'])
                df5 = pd.concat([df22, df44], ignore_index=True).copy()
                df52 = df5.replace({'type': 9}, 'antibiotics')
                df52 = df51.replace({'type': 4}, 'symptoms')
                df52.sort_values(['date'], inplace=True)
                df52.to_csv(path_out.format(file_name + '.csv'),header=True, index=False)  #경로2
            else:
                #pass
                df5 = pd.concat([df22, df35, df44], ignore_index=True).copy()
                df51 = df5.replace({'type': 9}, 'antibiotics')
                df52 = df51.replace({'type': 4}, 'symptoms')
                df52.sort_values(['date'], inplace=True)
                #print(df52)
                df52.to_csv(path_out.format(file_name + '.csv'),header=True, index=False)  #경로2
        except:
            discarded_file.append([file_name, 'unknown_error_try'])

#%%

#print(discarded_file)
listToDF = pd.DataFrame(discarded_file, columns=["finename", 'reason'])
listToDF.to_csv(path_save.format('discarded_file.csv'), header=True, index=False)  #경로2

#%%

c1=collections.Counter(weekNumList)
#plt.bar(c1.keys(), counter.values())
c2 = dict(c1)
c3 = sorted(c2.items())
df6 = pd.DataFrame(c3, columns=['weekNum', 'value'])
r = 0
df6['surveillance'] = float(0)
for p, q in c3:
    surveillance = (int(q)/int(len(weekNumList)))*100
    #print("surveillance in week " + str(p) + " : " + str(surveillance))
    df6['surveillance'][r] = surveillance
    r=r+1
#print(df6)
#np.save(path_save.format("surveillance"), c3)  #경로2
df6.to_csv(path_save.format("surveillance.csv"), header=True, index=False)  #경로2

#%%

#refdf = pd.read_csv(r'C:\Users\admin\Documents\fevercoach\from_new_world\test\medical_bigdata\surveillance.csv')
refdf = pd.read_csv(path_save.format("surveillance.csv"))
#print(refdf)
week_num = refdf['weekNum'].tolist()
for i in range(52):
    if i+1 not in week_num:
        refdf.loc[len(refdf)+1] = [i+1, 0, 0]
refdf.to_csv(path_save.format("surveillance.csv"), header=True, index=False)
