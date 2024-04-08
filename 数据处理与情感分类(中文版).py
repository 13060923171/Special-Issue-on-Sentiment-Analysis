import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from transformers import pipeline

# 去掉重复行以及空值
df1 = pd.read_csv('data.csv')
df2 = pd.read_csv('data1.csv')
df = pd.concat([df1,df2],axis=0)
df.drop_duplicates(subset=['content'], inplace=True)

# 导入停用词列表
stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


# 定义机械压缩函数
def yasuo(st):
    for i in range(1, int(len(st) / 2) + 1):
        for j in range(len(st)):
            if st[j:j + i] == st[j + i:j + 2 * i]:
                k = j + i
                while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
                    k = k + i
                st = st[:j] + st[k:]
    return st


def get_cut_words(content_series):
    try:
        # 对文本进行分词和词性标注
        words = pseg.cut(content_series)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            if flag.startswith('n') or flag.startswith('a'):
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


df['content'] = df['content'].apply(yasuo)
df['分词'] = df['content'].apply(get_cut_words)
new_df = df.dropna(subset=['分词'], axis=0)


classifier = pipeline('sentiment-analysis')
label_list = []
score_list = []
for d in new_df['content']:
    class1 = classifier(d)
    label = class1[0]['label']
    score = class1[0]['score']
    if score <= 0.55:
        label = 'NEUTRAL'
        label_list.append(label)
    else:
        label = label
        label_list.append(label)
    score_list.append(score)

new_df['情感类型'] = label_list
new_df['情感得分'] = score_list

new_df.to_csv('new_data.csv', encoding='utf-8-sig', index=False)

