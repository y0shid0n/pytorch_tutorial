from transformers import pipeline
from transformers import AlbertTokenizer
from transformers import BertForMaskedLM

model_dir = "./model/"

# futurewarning出てる。deprecated
tokenizer = AlbertTokenizer.from_pretrained(f'{model_dir}SousekiSP/souseki_sentencepiece.model', keep_accents=True)
model = BertForMaskedLM.from_pretrained(f'{model_dir}SousekiBERT')

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

MASK_TOKEN = tokenizer.mask_token

text = '''
吾輩は{}である。名前はまだ無い。
'''.format(MASK_TOKEN)

fill_mask(text)
