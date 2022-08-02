from multiprocessing.dummy import current_process
## 윈도우 : txt
## MAC : csv 
## txt to csv 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import csv

def txt_to_csv(txt_file):
    ## txt 파일 불러오기 및 데이터프레임으로 변경
    df = pd.DataFrame({
        'date':[], 
        'week':[],
        'time':[],
        'user':[],
        'message':[],
        })

    reader = txt_file.split('\n')
    date_ = ''
    week_ = ''
    time_ = ''
    user_ = ''
    message_ = ''

    for i in reader:
        
        txt1 = str(''.join(i))
        print(txt1)
        
        

        # 비어있는 문장 pass
        if len(txt1) > 1 :
            # 정상 문장 체크 ex. "[이태훈] [오후 5:02] 안녕하세요~!" 
            if txt1[0] == '[' and txt1.count('[') >= 2 and txt1.count(']') >= 2:
                ## user 설정
                user_ = txt1[txt1.find('['):txt1.find(']')][1:].strip()
                #print(user_)

                ## time 설정
                time_raw = txt1[txt1.find('[', txt1.find(']')):txt1.find(']',len(user_)+2)][1:]
                am_pm_split = time_raw[:time_raw.find(' ')].strip()
                time_split = time_raw[time_raw.find(' '):].strip()
                if am_pm_split == '오후':
                    time_hour = str(int(time_split[:time_split.find(':')].strip())+12)
                    time_minute = time_split[time_split.find(':'):].strip()
                    time_split = time_hour + time_minute
                time_ = time_split
                #print(am_pm_split,'/',time_split,'/',txt1)
                
                ## message 설정
                message_ = txt1[txt1.rfind(']')+1:].strip()
                #print(message_ ,'/',txt1)

            else: # 비정상 문장
                ## date 설정 
                if txt1.find('---------------') != -1 and txt1.rfind('---------------') != -1 :
                    date_raw = txt1[txt1.find(' '): txt1.rfind(' ')].strip() 
                    if date_raw.find('년') != -1 and date_raw.find('월') != -1 and date_raw.find('일') != -1: # 년 월 일 포함한다면
                        date_ = date_raw[:date_raw.rfind(' ')].replace('년 ','-').replace('월 ','-').replace('일','').strip()
                        week_ = date_raw[date_raw.rfind(' '):]
                        #print(date_raw,'/',date_, '/',week_)
                


        # 추가할 데이터 
        data_to_insert = {
            'date': date_, 
            'week': week_,
            'time': time_, 
            'user': user_,
            'message': message_,
        }
            
        # 데이터 추가해서 원래 데이터프레임에 저장하기
        if data_to_insert['date'] != '' and data_to_insert['week'] != '' and \
            data_to_insert['time'] != '' and data_to_insert['user'] != '' and \
                data_to_insert['message'] != '':
            df = df.append(data_to_insert, ignore_index=True)
            #df = pd.concat([df, pd.DataFrame(data_to_insert)],ignore_index=True)

            message_ = ''


    # 결측치 있으면 행 제거
    df.dropna()

    # csv 파일로 저장
    #new_csv_file = df.to_csv('py_01_kakao_csv.csv')
    return df