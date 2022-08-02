import py_01_txt_to_csv
import py_02_no_message_eda

import pandas as pd
import csv

# 0. load txt 파일 불러오기
# f = open('input_kakao.txt', encoding='utf8')
# txt_file = csv.reader(f)


# 1. txt to csv (텍스트 파일 csv파일로 변환하기)
def def_01_txt_to_csv(txt_file):
    df1 = py_01_txt_to_csv.txt_to_csv(txt_file)
    return df1

# 2. add_columns (컬럼 추가 및 정리)
def def_02_add_columns(df1):
    df2 = py_02_no_message_eda.add_columns(df1)
    return df2
#df2 = py_02_eda.add_columns('py_01_kakao_csv.csv')

# 3. No message eda 
def def_03_eda(df2): # df2 == py_02_kakao_csv.csv
    no_message_report_dict = py_02_no_message_eda.eda(df2)
    return no_message_report_dict


# 테스트 코드 
# df999 = pd.read_csv('py_02_kakao_csv.csv')
# no_message_report_dict = py_02_no_message_eda.eda(df999)
# print(no_message_report_dict)


# # load csv 
# load_csv_file = pd.read_csv('kakao_csv.csv')
# load_csv_file2 = load_csv_file.iloc[:,1:]
# print(load_csv_file2.head())

