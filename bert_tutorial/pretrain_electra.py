from transformers import AlbertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers import ElectraConfig, ElectraForMaskedLM
from datasets import load_dataset
import torch

txt_dir = "./data/"
model_dir = "./model/"
vocab_size = 32000

tokenizer = AlbertTokenizer.from_pretrained(f'{model_dir}SousekiSP/souseki_sentencepiece.model', keep_accents=True)

print(tokenizer.convert_ids_to_tokens(tokenizer("吾輩は猫である")['input_ids']))
print(tokenizer.vocab_size)
print(tokenizer.all_special_tokens)

config = ElectraConfig(vocab_size=vocab_size+3)
model = ElectraForMaskedLM(config)

# datasetsを使った読み込み（たぶんこれでいけるはず）
def tokenize_function(examples):
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=f"{model_dir}/SousekiELECTRA/",
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=48,
    save_steps=10000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
# 途中から再開する場合
# trainer.train(resume_from_checkpoint=True)

trainer.save_model(f'{model_dir}SousekiELECTRA/')

# メモリを空ける
del trainer, dataset, model
torch.cuda.empty_cache()
