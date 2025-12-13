from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch

from sklearn.metrics import classification_report

import numpy as np
import random
import time
import datetime
import os

import wandb

from utils import check_gpu, format_time, flat_accuracy
from dataloader import Dataset


class Models():
    def __init__(self, model_name='krbert', num_labels='2', dataset_instance=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = check_gpu()
        
        # 외부 Dataset 인스턴스 사용
        if dataset_instance is None:
             # 임시 Dataset 인스턴스만 생성하고 오류를 방지하기 위해 토크나이저만 기본값으로 설정
             self.dataset = Dataset()
             self.tokenizer = self.dataset.tokenizer
             # NOTE: 추론 시 반드시 dataset_instance가 명시적으로 전달되어야 합니다.
        else:
            self.dataset = dataset_instance
            self.tokenizer = self.dataset.tokenizer

    def BERT(self):
        if self.model_name == 'bert':       
            self.model = model = BertForSequenceClassification.from_pretrained(
                                        "bert-base-uncased",    # Use the 12-layer BERT model, with an uncased vocab.
                                        num_labels = self.num_labels,  # The number of output labels--2 for binary classification.
                                        output_attentions = False,  # Whether the model returns attentions weights.
                                        output_hidden_states = False,   # Whether the model returns all hidden-states.
                                        return_dict = False,
                                    )
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        elif self.model_name == 'krbert':
            self.model = model = BertForSequenceClassification.from_pretrained(
                                        "snunlp/KR-BERT-char16424",    # Use the 12-layer BERT model, with an uncased vocab.
                                        num_labels = self.num_labels,  # The number of output labels--2 for binary classification.
                                        output_attentions = False,  # Whether the model returns attentions weights.
                                        output_hidden_states = False,   # Whether the model returns all hidden-states.
                                        return_dict = False,
                                    )
            tokenizer = BertTokenizer.from_pretrained("snunlp/KR-BERT-char16424")

        self.model = model.to(self.device)

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                            lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                            eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                          )
        return tokenizer 

    def about_model(self):
        # Get all of the model's parameters as a list of tuples.
        params = list(self.model.named_parameters())

        print(f'The {self.model_name.upper()} model has {len(params)} different named parameters.\n')

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


    def train(self, train_dataloader, validation_dataloader, epochs = 4, project_title=None, project_entity=None):
        # after login to wandb at shell command
        if project_title:
            wandb.init(project=project_title, entity=project_entity)

        # Number of training epochs. The BERT authors recommend between 2 and 4. 
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.

        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # ========================================
            #             Training
            # ========================================
            
            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 50 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: token_type_ids
                #   [3]: labels 
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_token_type_ids = batch[2].to(self.device)
                b_labels = batch[3].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = self.model(b_input_ids, 
                                        # token_type_ids=b_token_type_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask, 
                                        labels=b_labels)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()
                
                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                self.optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)           
            
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
                
            # ========================================
            #             Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: token_type_ids
                #   [3]: labels 
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_token_type_ids = batch[2].to(self.device)
                b_labels = batch[3].to(self.device)
                
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits) = self.model(b_input_ids, 
                                            token_type_ids=b_token_type_ids, 
                    #                         token_type_ids=None, 
                                            attention_mask=b_input_mask,
                                            labels=b_labels)
                    
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
            # add to log training_stats in wandb
            if project_title:
                wandb.log(training_stats[epoch_i])

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    def test(self, test_dataloader):
        # Prediction on test set (전체 데이터셋 테스트 로직)
        print('Predicting labels for {:,} total test sentences...'.format(len(test_dataloader)))

        # -------------------------------------------------------------------
        # [수정] 1. 전체 데이터셋 테스트 로직은 그대로 유지 (test_dataloader를 사용)
        # -------------------------------------------------------------------
        
        self.model.eval()
        predictions , true_labels = [], []
        
        # 전체 데이터셋 예측 수행 (기존 로직)
        for batch in test_dataloader:
            # ... (기존 예측 로직 그대로)
            b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
            with torch.no_grad():
                outputs = self.model(input_ids = b_input_ids, token_type_ids = b_token_type_ids, attention_mask = b_input_mask)
            logits = outputs[0].detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()

            for logit in logits:
                predictions.append(np.argmax(logit))
            for ids in label_ids:
                true_labels.append(ids)

        print('\n--- [전체 데이터셋] 테스트 결과 ---')
        target_names = ['비도덕성 없음 (0)', '비도덕성 있음 (1)']
        print(classification_report(true_labels, predictions, target_names=target_names))
        print('    DONE (Total Test).\n')


        # -------------------------------------------------------------------
        # [추가] 2. 유형별 테스트 로직 추가
        # self.dataset 인스턴스를 사용하여 유형별 DataLoader를 가져옵니다.
        # -------------------------------------------------------------------
        print('\n--- [유형별 상세 테스트 결과] ---')
        
        if hasattr(self.dataset, 'get_dataloader_by_type'):
            # 다중 유형을 포함하도록 수정된 메서드를 호출
            type_dataloaders = self.dataset.get_dataloader_by_type(select='test')

            if not type_dataloaders:
                print("유형별 분석을 위한 유효한 데이터가 없습니다.")
                return

            target_names = ['비도덕성 없음 (0)', '비도덕성 있음 (1)']
            self.model.eval() # 모델을 평가 모드로 설정 (반복 불필요)
            
            for type_name, dataloader in type_dataloaders.items():
                print(f'\n--- [유형별 테스트 결과] 유형: {type_name} ---')
                
                type_predictions, type_true_labels = [], []
                
                # 유형별 데이터셋 예측 수행
                for batch in dataloader:
                    # 레이블이 있다고 가정하고 4개의 텐서를 언팩
                    if len(batch) != 4:
                        print(f"Error: 유형 '{type_name}'의 dataloader는 레이블을 포함해야 합니다.")
                        continue

                    b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch

                    with torch.no_grad():
                        outputs = self.model(input_ids = b_input_ids, token_type_ids = b_token_type_ids, attention_mask = b_input_mask)
                    
                    logits = outputs[0].detach().cpu().numpy()
                    label_ids = b_labels.cpu().numpy()

                    # 예측 및 실제 레이블 저장
                    type_predictions.extend(np.argmax(logit) for logit in logits)
                    type_true_labels.extend(label_ids)
                        
                if type_true_labels:
                    # 유형별 분류 리포트 출력
                    print(f"총 샘플 수: {len(type_true_labels):,}")
                    print(classification_report(type_true_labels, type_predictions, target_names=target_names, zero_division=0))
                else:
                    print("해당 유형에 유효한 레이블 데이터가 없어 리포트를 생성할 수 없습니다.")
        
        print('\n--- 유형별 상세 테스트 종료 ---')

    def inference(self, sentence):
        # Prediction on test set

        # print(f'Predicting labels for {sentence}')

        dataloader_ = self.dataset.get_infer_dataloader(data = [sentence])

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables 
        predictions = []

        # Predict 
        for batch in dataloader_:
            # Add batch to GPU
            # batch = tuple(t.to(device) for t in batch)
            # print(batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_token_type_ids = batch

            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(input_ids = b_input_ids, 
                                token_type_ids = b_token_type_ids, 
                                attention_mask = b_input_mask,
                                )

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            for logit in logits:
                pred = np.argmax(logit)
                predictions.append(pred)

        # print('    DONE.')
        # print(f"'{sentence}' 은/는 폭력성이 포함된 문장입니다" if predictions[0] == 1 else f"'{sentence}' 은/는 폭력성이 포함되지 않은 문장입니다")

        # return f"'{sentence}' 은/는 폭력성이 포함된 문장입니다" if predictions[0] == 1 else f"'{sentence}' 은/는 폭력성이 포함되지 않은 문장입니다"
        return sentence, predictions[0]


    # save_dir_path를 인수로 받도록 수정
    def save_model(self, save_dir_path):
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

        # save_dir = filedialog.askdirectory(initialdir = "/", title = "Please select a model save directory")
        save_dir = save_dir_path

        # Create output directory if needed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Saving model to {save_dir}")

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


    # load_dir_path를 인수로 받도록 수정
    def load_model(self, load_dir_path):
        self.BERT()

        # load_dir = filedialog.askdirectory(initialdir = "/", title = "Please select a model load directory")
        load_dir = load_dir_path
        print(f"Loading model from {load_dir}")
        
        # Load a trained model and vocabulary that you have fine-tuned
        self.model = self.model.from_pretrained(load_dir)
        self.tokenizer = self.tokenizer.from_pretrained(load_dir)

        # Copy the model to the GPU.
        self.model.to(self.device)

        return self.tokenizer