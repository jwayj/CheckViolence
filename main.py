from flask import Flask, render_template, request

from dataloader import Dataset
from model import Models
from transformers import BertTokenizer

app = Flask(__name__)

@app.route('/')
@app.route('/isViolence')
def Predict():
    input_text = request.args.get('input_text')
    
    print("input text : ", input_text)
    
    if input_text == None:
        return render_template('isViolence.html', Output = '')

    ModelOutput = model.inference(input_text)
    
    print("output : ", ModelOutput)

    return render_template('isViolence.html', Output = ModelOutput)

if __name__ == '__main__':

    # 모델 생성, 학습, 모델 저장
    model = Models('krbert', num_labels = 2)
    dataset = Dataset()
    dataset.set_dataset('train')    # train dataset load
    dataset.set_dataset('test')     # test dataset load
    dataset.set_tokenizer(BertTokenizer.from_pretrained("snunlp/KR-BERT-char16424"))
    print(dataset.get_tokenizer())
    train, valid, test = dataset.get_dataloader()   # train/valid/test dataloader 생성
    model.BERT()    # 모델 생성
    model.about_model()
    model.train(train, valid, epochs = 1)    # 학습
    model.test(test)    # 테스트
    model.save_model()

    # # 모델 생성, 모델 불러오기, 문장 폭력성 여부 추론
    # model = Models(num_labels = 2)
    # model.load_model()
    # app.run(host='0.0.0.0', port=5000)
    # # model.inference('이런건 걍 돈없고 멍청한 전라도 여자들 겨냥한 상술이지')

