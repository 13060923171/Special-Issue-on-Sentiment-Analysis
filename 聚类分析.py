import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn import metrics
import os
from sklearn.metrics import silhouette_score


sns.set_style(style="whitegrid")


def kmeans():
    corpus = []
    df = pd.read_csv('new_data.csv',encoding='utf-8-sig')
    # 读取预料 一行预料为一个文档
    for d in df['分词']:
        corpus.append(d.strip('\n'))

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names_out()

    # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    silhouette_scores = []
    best_k = None
    # 设置聚类数量K的范围
    range_n_clusters = range(2, 11)
    # 计算每个K值对应的轮廓系数
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(weight)
        score = silhouette_score(weight, labels)
        silhouette_scores.append(score)

        if best_k is None or score > silhouette_scores[best_k - 2]:
            best_k = n_clusters

    # Print the best K value and its corresponding silhouette score
    print(f"Best K value: {best_k}")
    print(f"Silhouette score for best K value: {silhouette_scores[best_k - 2]}")

    # 绘制轮廓系数图
    plt.plot(range_n_clusters, silhouette_scores, 'bo-', alpha=0.8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for K-means clustering')
    plt.savefig('轮廓系数图.png')
    plt.show()

    # n_clusters = best_k
    n_clusters = 3
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    # 使用PCA对特征进行降维
    pca = decomposition.PCA(n_components=2)
    pca.fit(weight)
    pca_vectors = pca.transform(weight)

    clf = KMeans(n_clusters=n_clusters, random_state=111)
    pre = clf.fit_predict(pca_vectors)

    x = [n[0] for n in pca_vectors]
    y = [n[1] for n in pca_vectors]
    plt.figure(figsize=(12, 9), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x, y, c=pre, s=100)
    plt.title("聚类分析图")
    plt.savefig('聚类分析图.png')
    plt.show()

    df['聚类结果'] = list(pre)
    df.to_csv('聚类结果.csv', encoding="utf-8-sig",index=False)


if __name__ == '__main__':
    kmeans()
