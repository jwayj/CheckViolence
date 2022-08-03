import torch
import numpy as np
import datetime
import re


def check_gpu():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def split_sentence(chattings):
    idx = 0
    result = []
    flag = False
    while idx < len(chattings):
        # '--------------- 2020년 9월 13일 일요일 ---------------' 같은 시간 문자열은 continue
        if re.findall(r'[-]+ [\d]+년 [\d]+월 [\d]+일 [가-힣]요일 [-]+', chattings[idx]):
            # print("continue : ", chattings[idx])
            idx += 1
            continue
        
        if not chattings[idx]:
            idx += 1
            continue

        if flag: # 채팅 내용 시작 후
            if chattings[idx][0] == '[': # [이름] [시간] 형식으로 되어 있는 문장인 경우
                name_time = re.findall(r'\[[^\]]*\] \[[^\]]*\]', chattings[idx])

                try:
                    name, time = re.findall(r'\[[^\]]*\]', name_time[0])
                except:
                    # print("except됨")
                    # print("index : ", idx)
                    # print("문장 : ", chattings[idx])
                    result[-1][2] += ' ' + chattings[idx].replace('\n', '')

                sentence = re.split(r'\[[^\]]*\] \[[^\]]*\]', chattings[idx])[-1][1:].replace('\n', '')
                result.append([name[1:-1], time[1:-1], sentence])

            else: # 이전 채팅에 붙어야 하는 문장인 경우
                result[-1][2] += ' ' + chattings[idx].replace('\n', '')
                
        
        else: # 채팅 내용 시작 전
            if chattings[idx][0] == '[':
                flag = True
                name_time = re.findall(r'\[[^\]]*\] \[[^\]]*\]', chattings[idx])

                try:
                    name, time = re.findall(r'\[[^\]]*\]', name_time[0])
                except:
                    # print("except됨")
                    # print("index : ", idx)
                    # print("문장 : ", chattings[idx])
                    result[-1][2] += ' ' + chattings[idx].replace('\n', '')

                sentence = re.split(r'\[[^\]]*\] \[[^\]]*\]', chattings[idx])[-1][1:].replace('\n', '')
                result.append([name[1:-1], time[1:-1], sentence])
            
        idx += 1

    return result