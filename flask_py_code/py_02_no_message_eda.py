import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pandasql as ps
import seaborn as sns
from matplotlib import font_manager, rc

'''
# None message eda 
01. 참여 사람들(users)
02. 참여 인원(user_count)
03. 대화 많은 사람(talk_count -> user , count)
04. 대화 기간(term -> min, max)
05. 대화가 가장 많은 월(mm)
06. 대화가 가장 많은 일(dd)
07. 대화가 가장 많은 요일(ww)
08. 대화가 가장 많은 시간대(hh)
총 채팅 횟수
# Mesaage eda 
분석한 키워드 수(message)
이모티콘
사진
'''
def add_columns(df1): # df1 == kakao_csv.csv
    #tt = pd.read_csv(df1)
    tt = df1

    # Date 파생 변수
    tt['date'] = pd.to_datetime(tt['date'])
    tt['Wday'] = tt['date'].dt.day_name()
    tt['yyyymm'] = tt['date'].dt.strftime('%Y%m')
    tt['yyyy'] = tt['date'].dt.strftime('%Y')
    tt['mm'] = tt['date'].dt.strftime('%m')
    tt['dd'] = tt['date'].dt.strftime('%d')

    # 시간대
    title_time = []
    for i in range(len(tt)):
        ttl2 = tt['time'][i][:2]
        if ttl2.find(':') == -1: # 존재하지 않을 때
            title_time.append(ttl2)
        else:
            title_time.append(ttl2.replace(':',""))
    tt['hh'] = title_time 

    # 메시지 길이
    title_len =  []
    for i in range(len(tt)):
        ttl = len(tt['message'][i])
        title_len.append(ttl)
    tt['Mlen'] = title_len 


    #new_csv_file = tt.to_csv('py_02_kakao_csv.csv')
    return tt

def eda(df2):
    df999 = df2
    report_dict = {
        'users' : ','.join(df999['user'].value_counts().index),
        'usercount' : len(list(df999['user'].value_counts().index)),
        'talkcount' : {
            'user' : list(df999['user'].value_counts().index),
            'count': list(df999['user'].value_counts().apply(str).values),
        },
        'term' :str(df999['date'].min())[:str(df999['date'].min()).find(' ')] + ' ~ ' + str(df999['date'].max())[:str(df999['date'].max()).find(' ')],
        'mm' : mm_count(zip(df999['mm'].value_counts().index,df999['mm'].value_counts())),
        'dd' : dd_count(zip(df999['dd'].value_counts().index,df999['dd'].value_counts())),
        'hh' : hh_count(zip(df999['hh'].value_counts().index,df999['hh'].value_counts())),
        'ww' : ww_count(zip(df999['week'].value_counts().index,df999['week'].value_counts())),
        'talklength' : str(df999['date'].count())
    }

    return report_dict


def mm_count(df_mm):
    dict_ = {
        '1월' : 0,
        '2월' : 0,
        '3월' : 0,
        '4월' : 0,
        '5월' : 0,
        '6월' : 0,
        '7월' : 0,
        '8월' : 0,
        '9월' : 0,
        '10월' : 0,
        '11월' : 0,
        '12월' : 0,
    }
    for index,value in df_mm:
        if str(int(index))+'월' in dict_.keys():
            dict_[str(int(index))+'월'] = value

    return dict_

def dd_count(df_dd):
    dict_ = {
        '1일' : 0 ,'2일' : 0 ,'3일' : 0 ,'4일' : 0 ,'5일' : 0 ,'6일' : 0 ,'7일' : 0 ,'8일' : 0 ,'9일' : 0 ,
        '10일' : 0 ,'11일' : 0 ,'12일' : 0 ,'13일' : 0 ,'14일' : 0 ,'15일' : 0 ,'16일' : 0 ,'17일' : 0 ,'18일' : 0 ,'19일' : 0 ,
        '20일' : 0 ,'21일' : 0 ,'22일' : 0 ,'23일' : 0 ,'24일' : 0 ,'25일' : 0 ,'26일' : 0 ,'27일' : 0 ,'28일' : 0 ,'29일' : 0 ,
        '30일' : 0 ,'31일' : 0 ,'32일' : 0 ,
    }
    for index,value in df_dd:
        if str(int(index))+'일' in dict_.keys():
            dict_[str(int(index))+'일'] = value

    return dict_

def hh_count(df_hh):
    dict_ = {
        '1시' : 0 ,'2시' : 0 ,'3시' : 0 ,'4시' : 0 ,'5시' : 0 ,'6시' : 0 ,'7시' : 0 ,'8시' : 0 ,'9시' : 0 ,
        '10시' : 0 ,'11시' : 0 ,'12시' : 0 ,'13시' : 0 ,'14시' : 0 ,'15시' : 0 ,'16시' : 0 ,'17시' : 0 ,'18시' : 0 ,'19시' : 0 ,
        '20시' : 0 ,'21시' : 0 ,'22시' : 0 ,'23시' : 0 ,'24시' : 0 ,
    }
    for index,value in df_hh:
        if str(int(index))+'시' in dict_.keys():
            dict_[str(int(index))+'시'] = value

    return dict_

def ww_count(df_ww):
    dict_ = {
        '월요일' : 0,
        '화요일' : 0,
        '수요일' : 0,
        '목요일' : 0,
        '금요일' : 0,
        '토요일' : 0,
        '일요일' : 0,
    }
    for index,value in df_ww:
        dict_[index.strip()] = value

    return dict_
    