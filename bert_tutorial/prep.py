# https://qiita.com/m__k/items/6f71ab3eca64d98ec4fc

import re
# import os
# import numpy as np
# import pandas as pd
# from glob import glob
from tqdm import tqdm
from pathlib import Path

txt_dir = Path('./data/txt/')

file_list = txt_dir.glob("*.txt")

corpus = []
for filename in tqdm(file_list):
    with open(filename, 'r', encoding='shift-jis') as r:
        text = r.read()
    text = re.sub(r"《[^》]*》", "", text)
    text = text.replace("\u3000", "")
    head_num = 0
    tail_num = 0
    for sent in text.split('\n')[2:]:
        if '--' in sent:
            head_num += 1
            continue
        if head_num == 2:

            if sent == '':
                tail_num += 1
                if tail_num == 3:
                    break
                else:
                    continue
            else:
                tail_num = 0

            sent = re.sub(r'^［.*］$', '', sent)
            sent = re.sub(r'［.*］', '', sent)
            sent = re.sub(r'※［.*］', '', sent)
            if sent == '': continue
            sent = "。\n".join(sent.split('。'))
            for _s in sent.split('\n'):
                corpus.append(_s)

# NSP用に出力
with open('./data/corpus_nsp.txt', 'w', encoding='utf-8') as w:
    w.write("\n".join(corpus))

# 空行を除く
corpus = list(filter(("").__ne__, corpus))

with open('./data/corpus.txt', 'w', encoding='utf-8') as w:
    w.write("\n".join(corpus))
