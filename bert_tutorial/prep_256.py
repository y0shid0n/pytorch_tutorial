# 256トークンを超える文章を削る
# とりあえずその一文だけでなくセンテンス全体を削る（NSPのため）
import pandas as pd
from transformers import AlbertTokenizer

txt_dir = "./data/"
model_dir = "./model/"

with open(f"{txt_dir}corpus_nsp.txt", "r", encoding="utf-8") as f:
    corpus_nsp = f.readlines()

tokenizer = AlbertTokenizer.from_pretrained(f'{model_dir}SousekiSP/souseki_sentencepiece.model', keep_accents=True)

df_corpus_nsp = pd.DataFrame({"txt": corpus_nsp})
df_corpus_nsp["txt"] = df_corpus_nsp["txt"].str.replace("\n", "")
df_corpus_nsp["token"] = df_corpus_nsp["txt"].apply(lambda x: tokenizer.tokenize(x))
df_corpus_nsp["length_token"] = df_corpus_nsp["token"].apply(lambda x: len(x))

# 256以上のトークンを抽出
# 出てきた文章が含まれるセンテンスを目で見て消した
print(df_corpus_nsp.query("length_token > 256")[["txt", "length_token"]])
