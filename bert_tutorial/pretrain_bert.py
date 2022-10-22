# MLMのみの学習（NSPはやらない）

from transformers import AlbertTokenizer
from transformers import BertConfig
from transformers import BertForMaskedLM
# from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from sentencepiece import SentencePieceProcessor
from datasets import load_dataset
import torch

txt_dir = "./data/"
model_dir = "./model/"
vocab_size = 32000

# transformersでロードする
# AlbertTokenizerではkeep_accents=Trueを指定しないと濁点が除去されてしまいます。
# futurewarning出てる。deprecated？とりあえず放置
tokenizer = AlbertTokenizer.from_pretrained(f'{model_dir}SousekiSP/souseki_sentencepiece.model', keep_accents=True)
text = "吾輩は猫である。名前はまだ無い。"
print(tokenizer.tokenize(text))
print(tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids']))

###########
# # sentencepieceでロードする（どっちでもよさげ？）
# # こっちはそのままbertには入れられない？
# sp = SentencePieceProcessor(model_file=f'{model_dir}SousekiSP/souseki_sentencepiece.model')

# # out_type=strを指定すれば分割文字列が確認できる
# print(sp.encode(text, out_type=str))
# # 指定しない場合はID
# print(sp.encode(text))

# # データの水増しもできる（確率的にtokenizeの結果を変える）
# for _ in range(3):
#     print(sp.SampleEncodeAsPieces(text, nbest_size=5, alpha=0.1))
###########

# BERTのconfigを定義
config = BertConfig(vocab_size=vocab_size+3, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
model = BertForMaskedLM(config)

# 学習
# ここでfuturewarning出てる（removeされるっぽい？）
# https://nikkie-ftnext.hatenablog.com/entry/replace-linebylinetextdataset-datasets-library
# dataset = LineByLineTextDataset(
#      tokenizer=tokenizer,
#      file_path=f'{txt_dir}corpus.txt',
#      block_size=256, # tokenizerのmax_length
# )

#####
# datasetsを使った読み込み（たぶんこれでいけるはず）
def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [
        line for line in examples["text"] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=256,  # from_pretrainedのmax_lengthと揃えた
        return_special_tokens_mask=True,
    )
# とりあえずsplitはしない
raw_datasets = load_dataset("text", data_files="./data/corpus.txt")
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=None,
    remove_columns=["text"],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset line_by_line",
)
dataset = tokenized_datasets["train"]
#####

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True,
    mlm_probability= 0.15
)

training_args = TrainingArguments(
    output_dir= f'{model_dir}SousekiBERTtest/',
    overwrite_output_dir=True,
    num_train_epochs=1,  # testは1でやる
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
# 途中から再開する場合
# trainer.train(resume_from_checkpoint=True)

trainer.save_model(f'{model_dir}SousekiBERTtest/')

# メモリを空ける
del trainer
torch.cuda.empty_cache()
