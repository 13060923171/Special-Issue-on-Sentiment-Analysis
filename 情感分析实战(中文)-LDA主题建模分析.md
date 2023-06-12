# 情感分析实战(中文)-LDA主题建模分析

**背景：该专栏的目的是将自己做了N个情感分析的毕业设计的一个总结版，不仅自己可以在这次总结中，把自己过往的一些经验进行归纳，梳理，巩固自己的知识从而进一步提升，而帮助各大广大学子们，在碰到情感分析的毕业设计时，提供一个好的处理思路，让广大学子们能顺利毕业**

## 情感分析实战：

a) [数据获取篇](https://blog.csdn.net/zyh960/article/details/131083616?spm=1001.2014.3001.5501)

b) [数据预处理篇-情感分类篇(中文版)](https://blog.csdn.net/zyh960/article/details/131083683?spm=1001.2014.3001.5501)

c) [数据预处理篇-情感分类篇(英文版)](https://blog.csdn.net/zyh960/article/details/131171641?spm=1001.2014.3001.5502)

d) [无监督学习机器学习聚类篇](https://blog.csdn.net/zyh960/article/details/131090242?spm=1001.2014.3001.5501)

e) [LDA主题分析篇](https://blog.csdn.net/zyh960/article/details/131092799?spm=1001.2014.3001.5501)

f) [共现语义网络](https://blog.csdn.net/zyh960/article/details/131095544?spm=1001.2014.3001.5502)

------



------



LDA(Latent Dirichlet Allocation)是一种主题模型，通常用于文本分析中。在聚类分析中，使用LDA的目的是对数据进行主题建模，从而捕捉数据的隐含特征，提高聚类效果。LDA能够根据数据内在结构，自动发现“话题”，对于探索文本主题或按照主题进行文本分类分析具有较强的实用价值。

LDA具有以下意义：

1. 挖掘数据内在结构：在聚类分析中，LDA可以通过挖掘样本数据背后的主题，识别隐藏于数据中的内在结构，从而能够更好地理解数据特征之间的相关性和相互关系。

2. 降低数据维度：在聚类分析中，为了避免过度拟合数据、提高模型泛化能力，通常需要对数据进行降维处理。LDA可以提取数据中的主题特征，从而降低数据维数，减少数据处理的复杂度。

3. 提高聚类精度：通过LDA建模分析，可以对文本数据进行更精细的划分，将数据划分为相似的主题类别。基于这种类别划分，可以建立更准确的聚类模型。LDA也可以基于主题相似性，自动进行聚类，从而提高聚类精度。

4. 数据可视化：LDA减小了高维度的数据量，更容易进行可视化建模，可使聚类结果更有可解释性。同时，LDA也可以通过可视化技术，可视化各类主题和其相关的文档，便于分析者进行数据挖掘、数据分析甚至决策。

通过LDA方法进行聚类分析，能够在一定程度上缓解高维数据的复杂度，提高聚类的准确性，进而增强聚类分析的可理解性、可解释性，具有良好的可视化效果。因此，在聚类分析中，采用LDA方法对数据进行主题建模，可以帮助我们更好的理解数据的内在结构和相关性，获得更高效和更精确的聚类结果。



**而在这个过程中，我们需要通过困惑度和一致性来确定我们的最优主题数**

困惑度（perplexity）和一致性（coherence）是选择最优LDA主题模型时常用的指标，通常使用困惑度和一致性来评估主题模型中隐含狄利克雷分布(Dirichlet Distribution)的质量和准确度。

困惑度主要用于评估主题模型对未标注语料库的拟合能力，即在该模型下，这些隐含主题与实际背景密切相关的程度。困惑度越小，表示模型对新语料库的拟合效果越好，有更好的预测效果。

一致性主要用于评估主题模型中被提取的主题的质量和稳定性，主要从主题词语（Topic Term）出现的频率和相关性来评估主题的一致性。一致性越高，表示这些词语在主题下存在更密切的相关性，反之则表明主题下的词汇较为松散，难以对主题进行概括。

使用困惑度、一致性的目的是为了选择最优LDA主题模型，从而获得更好的聚类效果和更高的可解释性。通过困惑度和一致性评估，可以比较不同的主题分布参数等模型参数，并选择效果最优的模型。

在实践中，对于LDA主题模型的选择，我们通常会使用不同的主题数，计算困惑度和一致性，并选择困惑度最小、一致性最高的主题数作为最佳模型参数，以获得更好的聚类效果和模型拟合度。

因此，通过困惑度、一致性指标的综合评估，可以帮助我们选择最佳的LDA主题模型，提高聚类效果和模型的可解释性。

具体代码如下：

```python
# 困惑度模块
x_data = []
y_data = []
z_data = []
for i in tqdm(range(2, 15)):
    x_data.append(i)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i)
    # 困惑度计算
    perplexity = lda_model.log_perplexity(corpus)
    y_data.append(perplexity)
    # 一致性计算
    coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    z_data.append(coherence)

# 绘制困惑度和一致性折线图
fig = plt.figure(figsize=(15, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 绘制困惑度折线图
ax1 = fig.add_subplot(1, 2, 1)
plt.plot(x_data, y_data, marker="o")
plt.title("perplexity_values")
plt.xlabel('num topics')
plt.ylabel('perplexity score')
# 绘制一致性的折线图
ax2 = fig.add_subplot(1, 2, 2)
plt.plot(x_data, z_data, marker="o")
plt.title("coherence_values")
plt.xlabel("num topics")
plt.ylabel("coherence score")

plt.savefig('./LDA主题/困惑度和一致性.png')
plt.show()
# 将上面获取的数据进行保存
df5 = pd.DataFrame()
df5['主题数'] = x_data
df5['困惑度'] = y_data
df5['一致性'] = z_data
df5.to_csv('./LDA主题/困惑度和一致性.csv', encoding='utf-8-sig', index=False)
```



呈现效果如图所示：

![困惑度和一致性1](https://cdn.jsdelivr.net/gh/13060923171/images@main/img/%E5%9B%B0%E6%83%91%E5%BA%A6%E5%92%8C%E4%B8%80%E8%87%B4%E6%80%A71.png)



一般而言，最佳主题数是由困惑度最小，一致性最高所对应的主题数确定的。根据提供的数据，可以看到困惑度在不同主题数下的变化趋势，以及随着主题数的增加，一致性的变化趋势。

通常情况下，困惑度会随着主题数的增加而减少，一致性则会随着主题数的增加先升高再下降。综合考虑困惑度和一致性指标，选择主题数使得困惑度最小、一致性最高，可以得到最优模型。

根据提供的数据，可以看到，主题数为4时困惑度最小，而主题数为6时一致性最高，因此，可以考虑选择该主题数(6)作为最佳主题数。但值得注意的是，选择最佳主题数时应该基于完整的数据集进行评估，并在保证较小的困惑度和较高的一致性的同时，尽量减小主题数，以获取更好的可解释性和模型准确性。



选择主题6的时候：

我们可以看出，每个主题分布都较为均匀

![主题强度1](https://cdn.jsdelivr.net/gh/13060923171/images@main/img/%E4%B8%BB%E9%A2%98%E5%BC%BA%E5%BA%A61.png)

去查看对应的LDA主题建模的时候：



![image-20230607171604269](https://cdn.jsdelivr.net/gh/13060923171/images@main/img/image-20230607171604269.png)

**每个气泡也分布较为均匀，对应这里的建模，这里解释一下**

LDA（Latent Dirichlet Allocation）气泡图是一种用于展示主题模型的可视化方式，通常用于表示文本或语料库的主题分布情况。LDA气泡图以一个二维平面上的主题分布图为基础，展示了不同主题之间的关系，主要由主题气泡和箭头构成。

气泡大小的代表意义：气泡大小通常代表LDA模型中的主题数量或主题词频，较大的气泡反映出该主题的权重较高，更重要或更具代表性。

气泡距离的代表意义：气泡之间的距离通常代表LDA模型中主题之间的相对距离关系。距离越近的主题，它们之间的主题词汇相关性越高，可能存在相近的主题标签或规律;反之，距离较远的主题之间可能存在更加明显的主题差异。

LDA气泡图通过多角度的呈现方式，可以帮助我们对LDA模型进行更加深入全面的分析。在可视化过程中，可以通过气泡的大小和距离，直观的感受到主题间的相对重要性、联系紧密程度等重要信息，进而进行可视化分析和解释，有助于更好地理解文本数据背后的主题结构。



这里有一点是需要注意的，因为LDA的HTML文件是需要建立在魔法上网的，如果用国内的网络是无法打开LDA.html这个文件的，所以我们需要更改一下这个文件里面的一些内容

把这三个文件放在同一个文件夹下面

![image-20230607172515714](https://cdn.jsdelivr.net/gh/13060923171/images@main/img/image-20230607172515714.png)

去到源代码里面进行修改，修改对应位置：

![image-20230607172634588](https://cdn.jsdelivr.net/gh/13060923171/images@main/img/image-20230607172634588.png)

![image-20230607172711898](https://cdn.jsdelivr.net/gh/13060923171/images@main/img/image-20230607172711898.png)

把这些相应的位置修改好过后，就可以正常的打开lda.html这个文件，呈现效果就和上面一样



接着我们来查看每个主题对应的权重词：

我们根据这些不同的主题的主题词，给对应的主题进行对应的主题归类，例如主题0,我们可以把主题0说成苹果手机方面的内容，主题1说成数码方面

| Topic0-主题词 | Topic0-权重 | Topic1-主题词 | Topic1-权重 | Topic2-主题词 | Topic2-权重 | Topic3-主题词 | Topic3-权重 | Topic4-主题词 | Topic4-权重 | Topic5-主题词 | Topic5-权重 |
| ------------- | ----------- | ------------- | ----------- | ------------- | ----------- | ------------- | ----------- | ------------- | ----------- | ------------- | ----------- |
| 音效          | 0.121       | 效果          | 0.15        | 不错          | 0.078       | 手机          | 0.124       | 京东          | 0.101       | 空调          | 0.084       |
| 屏幕          | 0.094       | 外形          | 0.094       | 质量          | 0.042       | 手感          | 0.068       | 苹果          | 0.045       | 师傅          | 0.069       |
| 外观          | 0.061       | 速度          | 0.074       | 物流          | 0.035       | 速度          | 0.065       | 不错          | 0.031       | 漂亮          | 0.046       |
| 流畅          | 0.049       | 待机时间      | 0.056       | 速度          | 0.029       | 系统          | 0.029       | 信赖          | 0.031       | 质感          | 0.032       |
| 苹果          | 0.04        | 外观          | 0.052       | 东西          | 0.027       | 屏幕          | 0.028       | 外观          | 0.027       | 专业          | 0.032       |
| 灵动          | 0.032       | 清晰          | 0.044       | 购物          | 0.024       | 外观          | 0.019       | 物流          | 0.027       | 格力          | 0.021       |
| 不错          | 0.027       | 屏幕          | 0.023       | 价格          | 0.021       | 京东          | 0.016       | 速度          | 0.025       | 奥克斯        | 0.02        |
| 效果          | 0.027       | 特色          | 0.017       | 京东          | 0.02        | 酸奶          | 0.015       | 品质          | 0.018       | 外观          | 0.02        |
| 像素          | 0.02        | 静音          | 0.015       | 鸡腿          | 0.016       | 紫色          | 0.013       | 时效          | 0.017       | 不错          | 0.018       |
| 特色          | 0.015       | 舒服          | 0.014       | 性价比        | 0.016       | 不错          | 0.013       | 下单          | 0.017       | 耐用          | 0.016       |
| 黑色          | 0.013       | 续航          | 0.014       | 味道          | 0.014       | 音质          | 0.013       | 老婆          | 0.016       | 素质          | 0.013       |
| 视频          | 0.013       | 不错          | 0.012       | 发货          | 0.013       | 大气          | 0.012       | 正品          | 0.015       | 降价          | 0.011       |
| 丝滑          | 0.011       | 冷暖          | 0.012       | 宝贝          | 0.013       | 发货          | 0.012       | 手机          | 0.014       | 很漂亮        | 0.011       |
| 信号          | 0.009       | 完美          | 0.012       | 很好          | 0.013       | 体验          | 0.012       | 舒服          | 0.014       | 外形          | 0.01        |
| 功能          | 0.009       | 手机          | 0.01        | 卖家          | 0.012       | 苹果          | 0.012       | 商品          | 0.014       | 耐心          | 0.009       |
| 细腻          | 0.009       | 空调          | 0.01        | 产品          | 0.012       | 颜色          | 0.012       | 外观设计      | 0.013       | 客服          | 0.009       |
| 电池          | 0.008       | 系统          | 0.008       | 品牌          | 0.01        | 习惯          | 0.011       | 品牌          | 0.012       | 速度          | 0.009       |
| 感觉          | 0.008       | 功能          | 0.008       | 仔细          | 0.009       | 很棒          | 0.01        | 体验          | 0.011       | 美的          | 0.009       |
| 很棒          | 0.008       | 品牌          | 0.007       | 感觉          | 0.009       | 白色          | 0.01        | 购物          | 0.011       | 时间          | 0.009       |
| 手机          | 0.008       | 白色          | 0.007       | 很棒          | 0.009       | 客服          | 0.01        | 外形          | 0.01        | 售后          | 0.009       |

针对这些主题数，我们的主要用法：

当我们使用LDA主题模型对文本进行聚类和主题建模时，得到的每个主题代表了一种语义主题或话题，并且包含了一组与该主题相关的单词，这些单词可以解释并描述该主题的含义。在实际应用中，LDA主题建模在以下方面具有重要的应用价值：

1. 文本分类：可以根据所得到的主题对文本进行分类，以帮助我们确定文本内容或主题

2. 推荐系统：可以将文档或文章作为“物品”，通过计算它们之间的主题相似度，使用主题相关性推荐相关的文章或文档。

3. 数据可视化: 通过LDA可视化工具（比如气泡图或者热图等），将文本数据可视化为在二维或多维空间中的主题或者主题分布，有助于我们了解数据的主题结构和特征，减轻分析瓶颈。

4. 信息检索：可以使用得到的主题模型来优化信息检索算法，提高搜索结果的准确性和覆盖率

5. 内容推荐：可以使用LDA模型对原始或标记好的文档进行分析建模，从而为用户推荐更有用、更相关的内容，提高用户满意度和平台活跃度。

总之，LDA主题建模方法为我们从海量的文本数据中提取出一个或多个主题，帮助人们更高效地理解、分析和利用文本数据。在实际应用过程中，LDA建模可以与其他数据分析技术结合使用，以实现更加优秀的数据分析效果。



总的代码：

```python
import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import itertools
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel


#LDA建模
def lda():
    df = pd.read_csv('./data/new_data.csv')
    train = []
    for line in df['分词']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]


    # 困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2, 15)):
        x_data.append(i)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=i)
        # 困惑度计算
        perplexity = lda_model.log_perplexity(corpus)
        y_data.append(perplexity)
        # 一致性计算
        coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        z_data.append(coherence)

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 绘制困惑度折线图
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x_data, y_data, marker="o")
    plt.title("perplexity_values")
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    #绘制一致性的折线图
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x_data, z_data, marker="o")
    plt.title("coherence_values")
    plt.xlabel("num topics")
    plt.ylabel("coherence score")

    plt.savefig('./LDA主题/困惑度和一致性.png')
    plt.show()
    #将上面获取的数据进行保存
    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('./LDA主题/困惑度和一致性.csv',encoding='utf-8-sig',index=False)
    num_topics = input('请输入主题数:')

    #LDA可视化模块
    #构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './LDA主题/lda.html')

    #主题判断模块
    list3 = []
    list2 = []
    #这里进行lda主题判断
    for i in lda.get_document_topics(corpus)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    data = pd.DataFrame()
    data['内容'] = df['分词']
    data['主题概率'] = list3
    data['主题类型'] = list2

    data.to_csv('./LDA主题/lda_data.csv',encoding='utf-8-sig',index=False)

    #获取对应主题出现的频次
    new_data = data['主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    #主题词模块
    word = lda.print_topics(num_words=20)
    topic = []
    quanzhong = []
    list_gailv = []
    list_gailv1 = []
    list_word = []
    #根据其对应的词，来获取其相应的权重
    for w in word:
        ci = str(w[1])
        c1 = re.compile('\*"(.*?)"')
        c2 = c1.findall(ci)
        list_word.append(c2)
        c3 = '、'.join(c2)

        c4 = re.compile(".*?(\d+).*?")
        c5 = c4.findall(ci)
        for c in c5[::1]:
            if c != "0":
                gailv = str(0) + '.' + str(c)
                list_gailv.append(gailv)
        list_gailv1.append(list_gailv)
        list_gailv = []
        zt = "Topic" + str(w[0])
        topic.append(zt)
        quanzhong.append(c3)

    #把上面权重的词计算好之后，进行保存为csv文件
    df2 = pd.DataFrame()
    for j,k,l in zip(topic,list_gailv1,list_word):
        df2['{}-主题词'.format(j)] = l
        df2['{}-权重'.format(j)] = k
    df2.to_csv('./LDA主题/主题词分布表.csv', encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('./LDA主题/特征词.csv',encoding='utf-8-sig',index=False)


#绘制主题强度饼图
def plt_pie():
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    df = pd.read_csv('./LDA主题/特征词.csv')
    x_data = list(df['所属主题'])
    y_data = list(df['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('主题强度')
    plt.tight_layout()
    plt.savefig('./LDA主题/主题强度.png')


if __name__ == '__main__':
    lda()
    plt_pie()

```



[整体项目地址](https://github.com/13060923171/Special-Issue-on-Sentiment-Analysis)

（小声bb：如果这个项目对你有用，不妨给我一个免费的小星星，非常感谢！！！）
