from sentencepiece import SentencePieceTrainer

txt_dir = "./data/"
model_dir = "./model/"
vocab_size = 32000

# Tokenizer
SentencePieceTrainer.Train(
    f'--input={txt_dir}corpus.txt, --model_prefix={model_dir}SousekiSP/souseki_sentencepiece --character_coverage=0.9995 --vocab_size={vocab_size} --pad_id=3 --add_dummy_prefix=False'
)
