# %%     import

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# import dask.dataframe as dd

import pandas as pd
import os

import glob
from tqdm import tqdm as tqdm  # 버전? 마다 출력 방식이 다름"
# from tqdm import tqdm
import time

# %%     define file path & name

input_root_folder = os.getcwd() + "\\data\\input\\"
output_root_folder = os.getcwd() + "\\data\\output\\"
raw_file_name = "\\data\\raw_mddata_20181231.csv"
raw_file_path = os.getcwd() + raw_file_name

pre1_folder = "pre1"
pre2_folder = "pre2"
pre3_folder = "pre3"
pre4_folder = "pre4"

flu_folder = "{0}\\{1}".format(pre2_folder, "flu")
odds_folder = "{0}\\{1}".format(pre2_folder, "odds")

csv_extension = ".csv"


# %% 폴더를 생성 함

def make_folder(folder_name):
    """
        폴더를 생성, ex) ./data/output/{folder_name}/
        :param folder_name : 폴더명

    """

    total_path = output_root_folder + "\\" + folder_name

    if not os.path.exists(total_path):
        print("make folder : " + total_path)
        os.makedirs(total_path)


# %%     RAW CSV 파일을 읽음

def load_raw_file():
    print("Start read raw csv file " + raw_file_path)
    start_time = time.time()
    #   df - 전역변수로 사용
    global df
    df = pd.read_csv(raw_file_path, low_memory=False)
    print("End read raw file : count : {0} / {1:0.2f} second".format(df.size, (time.time() - start_time)))


load_raw_file()


# %%     전처리 1

def pre_processing1():
    print("start pre_processing 1")
    make_folder(pre1_folder)
    output_folder = "{0}\\{1}\\".format(output_root_folder, pre1_folder)

    #   get id array
    id_array = df.loc[(df['type'] == 4) & (df['data_1'] == "3") & (df['data_2'] == "8"), 'baby_id'].unique().tolist()
    #   copy array
    copy_df = df[df['baby_id'].isin(id_array)].copy()
    #   sort by date
    copy_df.sort_values(by=['date'], inplace=True, ascending=True)  # inplace
    sorted_list = copy_df.loc[copy_df['baby_id'] != 0]

    for i, g in tqdm(sorted_list.groupby('baby_id')):
        file_name = output_folder + str(i) + csv_extension
        g.to_csv(file_name, header=True, index=False)

    print("end pre_processing 1")


pre_processing1()


# %%     전처리 2

def pre_processing2():
    print("start pre_processing2")

    diag = str(8)

    make_folder(flu_folder)
    make_folder(odds_folder)

    all_files = glob.glob(
        os.path.join("{0}\\{1}".format(output_root_folder, pre1_folder), "*{0}".format(csv_extension)))

    for file in tqdm(all_files):
        file_name = os.path.splitext(os.path.basename(file))[0]  # Getting the file name without extension
        file_df = pd.read_csv(file)  # Reading the file content to create a DataFrame
        file_df.index.name = file_name  # Setting the file name (without extension) as the index name

        if file_df.loc[
            (file_df['type'] == 4) & (file_df['data_1'] == "3")].empty:  # ��ü ����̶� Ȯ�� ���� ����� ���ԵǾ�����
            pass
        else:
            file_df['just_date'] = pd.to_datetime(file_df['date'], errors='coerce').dt.date
            file_df.loc[
                file_df['just_date'].shift(-1) - file_df['just_date'] > pd.Timedelta(value=1, unit='D'), 'c1'] = 2
            file_df.loc[
                file_df['just_date'].shift(-1) - file_df['just_date'] <= pd.Timedelta(value=1, unit='D'), 'c1'] = 0
            file_df.c1.iloc[[-1]] = 1
            file_df['c2'] = file_df.c1.cumsum()
            file_df['c3'] = file_df.apply(lambda x: x['c2'] - x['c1'], axis=1)
            grouped = file_df.groupby(['baby_id', 'c3'])  # ���̺���̵� �ʿ��Ѱ�?

            for k, g in grouped:
                index = 0
                #   pre 3 -> flu_true ?
                if not g.loc[(g['type'] == 4) & (g['data_1'] == "3") & (g['data_2'] == diag)].empty:
                    file_path = "{0}\\{1}\\{2}_{3}{4}".format(output_root_folder, flu_folder, file_name, str(index),
                                                              csv_extension)
                    g.to_csv(file_path, header=True, index_label=True)
                    # index+=1
                #   pre 3 -> flu_false ?
                elif not g.loc[(g['type'] == 4) & (g['data_1'] == "3") & (g['data_2'] != diag)].empty:
                    file_path = "{0}\\{1}\\{2}_{3}{4}".format(output_root_folder, odds_folder, file_name, str(index),
                                                              csv_extension)
                    g.to_csv(file_path, header=True, index_label=True)
                    # index+=1
    print("end pre_processing 2")


pre_processing2()


# %%     전처리 3

def pre_processing3():
    print("start pre_processing3")
    pd.options.mode.chained_assignment = None  # default='warn'
    class_translate = {"flu_true": 1, "flu_false": 0}

    print("end pre_processing3")


pre_processing3()

