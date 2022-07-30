from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from tkinter import filedialog, messagebox

from utils import check_gpu


class Dataset():
    def __init__(self):
        self.train = None
        self.valid = None
        self.test = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = 128
        self.device = check_gpu()
        print(f"max_length : {self.max_length}")

    def set_dataset(self, select = 'train'):
        if select == 'train':
            file = filedialog.askopenfilename(initialdir="/", 
                                        title = select + " 파일을 선택해주세요", 
                                        filetypes= (("*.csv","*csv"),("*.xlsx","*xlsx"),("*.xls","*xls")),
                                        )

            if file == '':
                messagebox.showwarning("경고", "파일을 선택해주세요")
            
            self.train, self.valid = train_test_split(pd.read_csv(file), test_size=0.3, random_state=42)
            print(f"Train : {len(self.train)} / Valid : {len(self.valid)}")
        
        elif select == 'test':
            file = filedialog.askopenfilename(initialdir="/", 
                                        title = select + " 파일을 선택해주세요", 
                                        filetypes= (("*.csv","*csv"),("*.xlsx","*xlsx"),("*.xls","*xls")),
                                        )

            if file == '':
                messagebox.showwarning("경고", "파일을 선택해주세요")
            
            self.test = pd.read_csv(file)
            print(f"Test : {len(self.test)}")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    # 입력받은 문장을 tokenizer를 통해 BERT input으로 만들어줌
    def make_input(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(sentence, \
                                                add_special_tokens = True,\
                                                pad_to_max_length=True,\
                                                max_length=self.max_length, 
                                                return_attention_mask=True,
                                                truncation = True,
                                                )

        return encoded_dict

    # 입력받은 문장 리스트를 BERT input에 적절한 dataset으로 만들어줌
    # labels를 함께 입력받으면 train, test용
    # labels 없이 입력받으면 inference용
    def make_dataset(self, sentences, labels = None):
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for line in tqdm(sentences):
        #     line = ' '.join(mecab.morphs(line)) # mecab 적용, encode하면 tokenizer.tokenize 해준 것과 같은 결과 나옴
            encoded_dict = self.make_input(line)

            input_id = encoded_dict['input_ids']
            attention_mask = encoded_dict['attention_mask']
            token_type_id = encoded_dict['token_type_ids']

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(self.device)
        # inputs = (input_ids, attention_masks, token_type_ids)

        # print("Original Text : ", sentences[0])
        # print("Tokenizer Text : ", BertTokenizer.from_pretrained("bert-base-uncased").tokenize(sentences[0]))
        # print("Encode Text : ", (BertTokenizer.from_pretrained("bert-base-uncased").encode(sentences[0], add_special_tokens = True, max_length = self.max_length)))
        print("Original Text : ", sentences[0])
        print("Tokenizer Text : ", self.tokenizer.tokenize(sentences[0]))
        print("Encode Text : ", (self.tokenizer.encode(sentences[0], add_special_tokens = True, max_length = self.max_length)))

        # labels = torch.tensor(df['violence'].tolist(), dtype=torch.long).to(device)
        # labels = torch.tensor(df.iloc[:, -1].tolist(), dtype=torch.long).to(self.device)

        if labels == None:
            return TensorDataset(input_ids, attention_masks, token_type_ids)
        else:
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            return TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    

    def get_dataloader(self):
        train_dataset = self.make_dataset(self.train.문장.tolist(), self.train.iloc[:, -1].tolist())
        valid_dataset = self.make_dataset(self.valid.문장.tolist(), self.valid.iloc[:, -1].tolist())
        test_dataset = self.make_dataset(self.test.문장.tolist(), self.test.iloc[:, -1].tolist())

        batch_size = 32

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            # sampler = RandomSampler(train_dataset),
            sampler = SequentialSampler(train_dataset),
            batch_size = batch_size,
        )
        validation_dataloader = DataLoader(
                    valid_dataset,
                    sampler = SequentialSampler(valid_dataset),
                    batch_size = batch_size,
                )

        test_dataloader = DataLoader(
                    test_dataset,
                    sampler = SequentialSampler(test_dataset),
                    batch_size = batch_size,
                )

        return train_dataloader, validation_dataloader, test_dataloader

    def get_infer_dataloader(self, data): # data : ['sentence1', 'sentence2', ...]
        dataset = self.make_dataset(data)

        batch_size = len(dataset)
        dataloader = DataLoader(
                    dataset,
                    sampler = SequentialSampler(dataset),
                    batch_size = batch_size,
                )

        return dataloader