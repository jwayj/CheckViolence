from flask import Flask, render_template, request

from dataloader import Dataset
from model import Models
from transformers import BertTokenizer
from utils import split_sentence

app = Flask(__name__)

@app.route('/')
def file_upload():
    return render_template('load_file.html')
    
@app.route('/isViolence', methods = ['POST', 'GET'])
def Predict():
    if request.method == 'POST':
        input_text = request.files['파일'].read().decode('utf-8') # 파일 불러오기
        input_text = input_text.split('\r\n')
        
        if input_text == None:
            return render_template('isViolence.html', Output = '')

        sentences = split_sentence(input_text)  # 채팅 목록을 문장 단위로 분리
        # print(sentences)
        result = 0
        for i, sentence in enumerate(sentences[:10000]):
            if (i+1) % 1000 == 0:
                print(f"{i+1}번째 실행 중...")
            result += model.inference(sentence[-1])[1]
        
        ModelOutput = f"해당 채팅방의 폭력성 대화 비율은 {round((result / len(sentences)) * 100, 2)}% 입니다"
        print(ModelOutput)
        return render_template('isViolence.html', Output = ModelOutput)

if __name__ == '__main__':

    # # 모델 생성, 학습, 모델 저장
    # dataset = Dataset()
    # dataset.set_dataset('train')    # train dataset load
    # dataset.set_dataset('test')     # test dataset load
    # dataset.set_tokenizer(BertTokenizer.from_pretrained("snunlp/KR-BERT-char16424"))
    # print(dataset.get_tokenizer())
    # model = Models('krbert', num_labels = 2)
    # train, valid, test = dataset.get_dataloader()   # train/valid/test dataloader 생성
    # model.BERT()    # 모델 생성
    # model.about_model()
    # model.train(train, valid, epochs = 1, project_title='wandb-test_NLP', project_entity='nlp-yd10')    # 학습
    # model.test(test)    # 테스트
    # model.save_model()

    # # 모델 생성, 모델 불러오기, 문장 폭력성 여부 추론
    # model = Models('krbert', num_labels = 2)
    # model.load_model()
    # sentence, prediction = model.inference('이런건 걍 돈없고 멍청한 전라도 여자들 겨냥한 상술이지')
    # print(f"'{sentence}' 은/는 폭력성이 포함된 문장입니다" if prediction == 1 else f"'{sentence}' 은/는 폭력성이 포함되지 않은 문장입니다")

    # 모델 생성, 모델 불러오기, 문장 폭력성 여부 추론 (웹에서)
    model = Models('bert', num_labels = 2)
    model.load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)

