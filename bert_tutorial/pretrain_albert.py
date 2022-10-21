from transformers import AlbertTokenizer, AlbertConfig, AlbertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers import LineByLineTextDataset
from datasets import load_dataset
import GPUtil
import torch

txt_dir = "./data/"
model_dir = "./model/"
vocab_size = 32000

tokenizer = AlbertTokenizer.from_pretrained(f'{model_dir}SousekiSP/souseki_sentencepiece.model', keep_accents=True)

print(tokenizer.convert_ids_to_tokens(tokenizer("吾輩は猫である")['input_ids']))
print(tokenizer.vocab_size)
print(tokenizer.all_special_tokens)

# config = AlbertConfig(vocab_size=vocab_size+3, embedding_size=256, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)
# num_attention_headsがよくないっぽい（よくわかってない）
config = AlbertConfig(vocab_size=vocab_size+3, embedding_size=256, intermediate_size=768)
model = AlbertForMaskedLM(config)

#####
# datasetsを使った読み込み（たぶんこれでいけるはず）
# def tokenize_function(examples):
#     # Remove empty lines
#     examples["text"] = [
#         line for line in examples["text"] if len(line) > 0 and not line.isspace()
#     ]
#     return tokenizer(
#         examples["text"],
#         padding=False,
#         truncation=True,
#         max_length=256,  # from_pretrainedのmax_lengthと揃えた
#         return_special_tokens_mask=True,
#     )
# # とりあえずsplitはしない
# raw_datasets = load_dataset("text", data_files="./data/corpus.txt")
# tokenized_datasets = raw_datasets.map(
#     tokenize_function,
#     batched=True,
#     num_proc=None,
#     remove_columns=["text"],
#     load_from_cache_file=True,
#     desc="Running tokenizer on dataset line_by_line",
# )
# dataset = tokenized_datasets["train"]
#####

# とりあえずこれで試す
dataset = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=f'{txt_dir}corpus.txt',
     block_size=256, # tokenizerのmax_length
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir= f'{model_dir}SousekiALBERT/',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# GPUのメモリが足らんぽい（途中でout of memoryで死ぬ）
trainer.train()
trainer.save_model(f'{model_dir}SousekiAlBERT/')
