from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
import torch
from tqdm import tqdm
# from sklearn.model_selection import train_test_split # <- train_test_split 제거
import pandas as pd

from utils import check_gpu

# ==============================================================================
# TODO: 여기에 실제 데이터 파일의 열 이름을 지정해주세요.
# 문장 텍스트가 들어있는 열 이름
TEXT_COLUMN_NAME = '문장' 
# 라벨(0 또는 1)이 들어있는 열 이름 (만약 마지막 열을 사용한다면 변경 불필요)
LABEL_COLUMN_NAME = '비도덕성 여부' # <- '비도덕성 여부' 열을 라벨로 명시적으로 지정
# ==============================================================================

# TARGET_DATA_SIZE 상수는 제거되었습니다. 이제 main.py에서 max_rows 인수를 사용합니다.

class Dataset():
    def __init__(self):
        self.train = None
        self.valid = None
        self.test = None
        self.tokenizer = BertTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
        self.max_length = 128
        self.device = check_gpu()
        print(f"max_length : {self.max_length}")

    # file_path는 단일 경로 또는 리스트, max_rows로 원하는 행 수를 지정합니다.
    def set_dataset(self, select='train', file_path=None, delimiter=',', max_rows=None):
        
        if file_path is None:
            raise ValueError(f"{select} dataset을 로드하려면 'file_path'를 지정해야 합니다.")

        try:
            # 파일 경로를 리스트 형태로 통일 (단일 경로도 리스트로 처리)
            files_to_load = [file_path] if isinstance(file_path, str) else file_path

            if not files_to_load:
                raise ValueError(f"{select} dataset을 로드할 파일 경로가 지정되지 않았습니다.")

            data_frames = []
            
            for f_path in files_to_load:
                # 파일 로드 시도
                try:
                    print(f"[{select}]: 로드 중 파일: {f_path}")
                    # 파일 확장자를 확인하여 적절한 판다스 함수로 읽기
                    if f_path.endswith('.csv'):
                        # max_rows가 지정된 경우, read_csv의 nrows 인수를 사용하여 처음부터 로드되는 양을 제한
                        # 단, 여러 파일을 합치는 경우 전체를 로드 후 샘플링하는 것이 더 정확함.
                        # 여기서는 일단 전체를 로드 후 샘플링하는 기존 로직을 따릅니다.
                        data = pd.read_csv(f_path, sep=delimiter)
                    elif f_path.endswith(('.xlsx', '.xls')):
                        data = pd.read_excel(f_path)
                    elif f_path.endswith('.json'):
                        data = pd.read_json(f_path, lines=True) 
                    else:
                        print(f"Warning: 지원되지 않는 파일 형식입니다: {f_path}. 해당 파일은 건너뜁니다.")
                        continue # 지원되지 않는 파일은 건너뛰기
                    
                    data_frames.append(data)
                except FileNotFoundError:
                    print(f"Warning: 파일을 찾을 수 없습니다: {f_path}. 해당 파일은 건너뜁니다.")
                except Exception as e:
                    print(f"Warning: 파일 {f_path} 로드 중 오류 발생: {e}. 해당 파일은 건너뜁니다.")


            # 모든 데이터 프레임을 하나로 합치기
            if not data_frames:
                 raise Exception(f"{select} 데이터셋을 로드하는 데 실패했습니다. 성공적으로 로드된 파일이 없습니다.")
                 
            data = pd.concat(data_frames, ignore_index=True)
            print(f"[{select}]: 총 {len(data_frames)}개의 파일에서 {len(data):,}개의 데이터를 성공적으로 병합했습니다.")

            # ------------------------------------------------------------------
            # 데이터 크기 줄이기 로직 (max_rows 인수를 사용)
            # ------------------------------------------------------------------
            if max_rows is not None and len(data) > max_rows:
                # 'n'을 max_rows로 지정하여 원하는 크기만큼 샘플링
                print(f"[{select}]: 원본 데이터 {len(data):,}개를 {max_rows:,}개로 샘플링합니다.")
                # 무작위 샘플링. random_state로 재현성 보장
                data = data.sample(n=max_rows, random_state=42)
            # 데이터 크기가 지정된 max_rows보다 작은 경우
            elif max_rows is not None and len(data) < max_rows: 
                print(f"[{select}]: 로드된 데이터 {len(data):,}개. 현재는 목표 크기({max_rows:,}개)보다 작으므로 샘플링을 건너뜁니다.")
            # ------------------------------------------------------------------


            if select == 'train':
                # train 데이터셋 전체를 파일에서 로드
                self.train = data
                print(f"Train : {len(self.train)}")
            
            elif select == 'valid':
                # valid 데이터셋을 파일에서 로드
                self.valid = data
                print(f"Valid : {len(self.valid)}")

            elif select == 'test':
                # test 데이터셋을 파일에서 로드
                self.test = data
                print(f"Test : {len(self.test)}")
            
            else:
                raise ValueError(f"'{select}'은(는) 유효한 데이터셋 유형이 아닙니다. ('train', 'valid', 'test' 중 선택)")

        except FileNotFoundError:
            # 이 경로는 files_to_load가 단일 경로일 때만 작동할 수 있으므로,
            # 위 개별 파일 로드 시 경고로 처리하는 것이 더 안전합니다.
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        except Exception as e:
            # 로드 실패 시 어떤 파일에서 문제가 발생했는지 출력하는 것이 좋습니다.
            raise Exception(f"데이터셋 로드 중 최종 오류 발생: {e}")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    # 입력받은 문장을 tokenizer를 통해 BERT input으로 만들어줌
    def make_input(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(sentence, \
                                                 add_special_tokens = True,\
                                                 padding='max_length',\
                                                 # pad_to_max_length=True,  <- 이 줄을 padding='max_length'로 대체
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
        # for line in tqdm(sentences):
        for line in sentences:
        #  line = ' '.join(mecab.morphs(line)) # mecab 적용, encode하면 tokenizer.tokenize 해준 것과 같은 결과 나옴
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
        # print("Tokenizer Text : ", self.tokenizer.tokenize(sentences[0]))
        # print("Encode Text : ", (self.tokenizer.encode(sentences[0], add_special_tokens = True, max_length = self.max_length)))

        if labels == None:
            return TensorDataset(input_ids, attention_masks, token_type_ids)
        else:
            
            def safe_to_int(label):
                if isinstance(label, bool):
                    return int(label)
                if isinstance(label, str):
                    lower_label = label.lower()
                    if lower_label == 'true':
                        return 1
                    elif lower_label == 'false':
                        return 0
                    try:
                        return int(label)
                    except ValueError:
                        print(f"Warning: Non-numeric/non-boolean string found: '{label}'. Defaulting to 0.")
                        return 0 # 안전을 위해 0으로 대체
                try:
                    return int(label)
                except:
                    print(f"Warning: Invalid label format found: '{label}'. Defaulting to 0.")
                    return 0

            labels = [safe_to_int(label) for label in labels] 
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            return TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    

    def get_dataloader(self):
        
        # ---------------------------------------------------------------------------------------
        # 라벨 추출 시 LABEL_COLUMN_NAME을 사용하여 '비도덕성 여부' 열을 사용하도록 설정
        # ---------------------------------------------------------------------------------------

        # 1. 문장 데이터 추출
        train_sentences = self.train[TEXT_COLUMN_NAME].tolist()
        valid_sentences = self.valid[TEXT_COLUMN_NAME].tolist()
        test_sentences = self.test[TEXT_COLUMN_NAME].tolist()

        # 2. 라벨 데이터 추출
        # LABEL_COLUMN_NAME이 '비도덕성 여부'로 설정되었으므로 해당 열을 사용
        train_labels = self.train[LABEL_COLUMN_NAME].tolist()
        valid_labels = self.valid[LABEL_COLUMN_NAME].tolist()
        test_labels = self.test[LABEL_COLUMN_NAME].tolist()
            
        train_dataset = self.make_dataset(train_sentences, train_labels)
        valid_dataset = self.make_dataset(valid_sentences, valid_labels)
        test_dataset = self.make_dataset(test_sentences, test_labels)

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

    def get_dataloader_by_type(self, select='test'):
        """
        선택된 데이터셋을 '유형' 컬럼의 쉼표(,)로 구분된 모든 유형을 기준으로 분리하여
        각 유형별 DataLoader 딕셔너리를 반환합니다.
        하나의 문장은 여러 유형의 데이터 로더에 포함될 수 있습니다.
        """
        if select == 'train':
            data = self.train
        elif select == 'valid':
            data = self.valid
        elif select == 'test':
            data = self.test
        else:
            raise ValueError("Select must be 'train', 'valid', or 'test'.")

        if data is None:
            print(f"Warning: {select} 데이터가 로드되지 않았습니다.")
            return {}

        # 모든 고유 유형을 추출 (쉼표로 분리하여 flat하게 만듦)
        all_types = data['유형'].astype(str).str.split(',').explode().str.strip().unique()
        
        type_dataloaders = {}
        batch_size = 32  # 기존 dataloader와 동일한 배치 크기 사용

        print(f"\n--- {select.upper()} 데이터를 모든 유형을 기준으로 분리 중... ---")

        for type_name in all_types:
            if type_name in ['', 'nan', 'IMMORAL_NONE']: # 유효하지 않거나 분석 대상이 아닌 유형 제외
                continue
                
            # 해당 유형을 포함하는 모든 행을 필터링합니다.
            # .str.contains()를 사용하여 해당 유형이 포함된 행을 찾습니다.
            # 정규식 패턴을 사용하여 정확히 해당 유형만 매칭되거나 쉼표로 분리된 경우를 처리합니다.
            # 예: 'ABUSE'를 포함하고 양쪽에 쉼표나 시작/끝 문자가 있는 경우
            # '유형' 컬럼을 문자열로 처리하고, NaN 값은 건너뜁니다.
            type_filter = data['유형'].astype(str).str.contains(r'\b' + type_name + r'\b')
            group = data[type_filter].copy()

            if group.empty:
                continue

            # 2. 각 유형별로 문장과 레이블 추출
            sentences = group['문장'].tolist()
            labels = group['비도덕성 여부'].tolist() # 레이블은 원본 데이터를 따릅니다.
            
            # 3. BERT 입력 형식의 TensorDataset 생성
            dataset_instance = self.make_dataset(sentences, labels)

            # 4. DataLoader 생성
            dataloader = DataLoader(
                dataset_instance,
                sampler=SequentialSampler(dataset_instance),
                batch_size=batch_size,
            )
            type_dataloaders[type_name] = dataloader
            print(f"  - 유형: {type_name:<20} | 샘플 수: {len(group):,}")
            
        return type_dataloaders

    def check_train_type_counts(self):
        """
        훈련 데이터셋(self.train)의 '유형' 컬럼을 분석하여 
        쉼표로 구분된 모든 유형별 개수를 출력합니다.
        """
        if self.train is None:
            print("❌ 훈련 데이터셋(self.train)이 로드되지 않았습니다.")
            return

        print("\n--- 훈련 데이터셋 (train) 유형별 개수 분석 ---")

        # 1. '유형' 컬럼을 쉼표(,)로 분리하고 모든 고유 유형을 추출
        # .astype(str)로 NaN을 'nan' 문자열로 처리하여 에러 방지
        # .explode()로 리스트를 행으로 펼칩니다.
        # .str.strip()으로 공백을 제거합니다.
        exploded_types = self.train['유형'].astype(str).str.split(',').explode().str.strip()
        
        # 2. 각 유형별 개수 계산
        type_counts = exploded_types.value_counts().sort_index()

        # 3. 결과 출력
        total_unique_sentences = len(self.train)
        print(f"**총 훈련 문장 수 (중복 제거 전): {total_unique_sentences:,}개**")
        print(f"**총 레이블 수 (다중 레이블 포함): {len(exploded_types):,}개**")
        print("-" * 40)
        
        # 각 유형별 개수 출력
        for type_name, count in type_counts.items():
            if type_name in ['', 'nan']: # 유효하지 않은 유형 제외
                 print(f"  - [{type_name.upper():<13}]: {count:,}개 (제외)")
                 continue

            print(f"  - {type_name:<13}: {count:,}개")
            
        print("-" * 40)

    def get_infer_dataloader(self, data): # data : ['sentence1', 'sentence2', ...]
        dataset = self.make_dataset(data)

        batch_size = len(dataset)
        dataloader = DataLoader(
                            dataset,
                            sampler = SequentialSampler(dataset),
                            batch_size = batch_size,
                        )

        return dataloader