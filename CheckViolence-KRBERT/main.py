from utils import check_gpu
from dataloader import Dataset
from model import Models

if __name__ == '__main__':

    # 모델 생성, 학습, 모델 저장
    model = Models(num_labels = 2)
    dataset = Dataset()
    dataset.set_dataset('train')    # train dataset load
    dataset.set_dataset('test')     # test dataset load
    train, valid, test = dataset.get_dataloader()   # train/valid/test dataloader 생성
    model.BERT()    # 모델 생성
    model.train(train, valid, epochs = 1)    # 학습
    model.test(test)    # 테스트
    model.save_model()

    # # 모델 생성, 모델 불러오기, 문장 폭력성 여부 추론
    # model = Models(num_labels = 2)
    # model.load_model()
    # model.inference('이런건 걍 돈없고 멍청한 전라도 여자들 겨냥한 상술이지')