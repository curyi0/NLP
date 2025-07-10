import  torch, pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
import evaluate
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments )
data= load_dataset('nsmc',trust_remote_code=True)
# with open("data/NLP_set.pkl" ,'wb') as f:
#     pickle.dump(data, f)
# exit()

#tokenizer 모델
# model_name="klue/roberta-large"
model_name="beomi/roberta-large"

tokenizer= AutoTokenizer.from_pretrained(model_name)

def tokenizer_func(example):
    token=AutoTokenizer.from_pretrained(model_name)
    return token(example["document"], padding="max_length", truncation=True)

def copute_metrics(eval_pred):
    accuracy_metric=evaluate.load("accuracy")
    prediction, labels= eval_pred
    prediction= np.argmax( prediction, axis=1)
    return accuracy_metric.compute(predictions=prediction, references=labels)

if __name__=="__main__":
    with open("venv/data/NLP_set.pkl", 'rb') as f:
        data=pickle.load(f)

    train=data["train"].shuffle(seed=42).select(range(1000))
    test=data["test"].shuffle(seed=42).select(range(500))

    # trainedDf= pd.DataFrame(train)
    tokenized_train_data=train.map(tokenizer_func, batched=True)
    tokenized_test_data=train.map(tokenizer_func, batched=True)
    tokenized_test_df=pd.DataFrame(tokenized_test_data)

    # print(tokenized_test_df.head())
    # label = target = 정답
    model=AutoModelForSequenceClassification(model_name, num_labels=2)

    n_epoch=3
    training_args= TrainingArguments(
        output_dir="venv/results", # 저장
        num_train_epochs=n_epoch,   # 훈련횟수
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay= 0.01,
        logging_dir="venv/logs",   #  학습 중 로그 기록
        logging_steps=10,
        eval_strategy= "epoch",  # 각 횟수 마다 평가 함
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    trainer= Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_df,
        copute_metrics=copute_metrics,
    )
    # print(accuracy_metric)

    trainer.train()
    save_directory= './save_model'
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)