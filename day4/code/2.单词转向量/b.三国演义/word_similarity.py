# 进行单词相似度匹配

from gensim.models import word2vec
import multiprocessing
import os

# 定义分词结果所在的文件夹路径
segment_folder = './segment'
# 使用 PathLineSentences 迭代读取文件夹下的所有文件作为语料，避免一次性加载到内存
sentences = word2vec.PathLineSentences(segment_folder)

# 训练第一个 Word2Vec 模型
# vector_size=100: 词向量维度为100
# window=3: 上下文窗口大小为3
# min_count=1: 词频少于1的词会被忽略
model = word2vec.Word2Vec(sentences, vector_size=100, window=3, min_count=1)

# 打印 '赵子龙' 的词向量
print(f"model.wv['赵子龙'] => {model.wv['赵子龙']}")
# 计算 '赵子龙' 和 '刘备' 的相似度
print(f"model.wv.similarity('赵子龙', '刘备') => {model.wv.similarity('赵子龙', '刘备')}")
# 计算 '赵子龙' 和 '赵云' 的相似度
print(f"model.wv.similarity('赵子龙', '赵云') => {model.wv.similarity('赵子龙', '赵云')}")
# 寻找与 '赵子龙' 和 '刘备' 最相似，但与 '赵云' 最不相似的词
print(f"model.wv.most_similar(positive=['赵子龙', '刘备'], negative=['赵云']) => {model.wv.most_similar(positive=['赵子龙', '刘备'], negative=['赵云'])}")

# 训练第二个 Word2Vec 模型，参数不同
# vector_size=128: 词向量维度为128
# window=5: 上下文窗口大小为5
# min_count=5: 词频少于5的词会被忽略
# workers=multiprocessing.cpu_count(): 使用多核CPU并行训练
my_model = word2vec.Word2Vec(sentences, vector_size=128, window=5, min_count=5, workers=multiprocessing.cpu_count())

# 确保模型保存目录存在
if not os.path.exists('./models'):
    os.makedirs('./models')

# 保存这个模型到指定路径
my_model.save('./models/三国演义相似度匹配.model')

# 计算 '赵子龙' 和 '刘备' 的相似度 (新模型)
print(f"my_model.wv.similarity('赵子龙', '刘备') => {my_model.wv.similarity('赵子龙', '刘备')}")
# 计算 '赵子龙' 和 '赵云' 的相似度 (新模型)
print(f"my_model.wv.similarity('赵子龙', '赵云') => {my_model.wv.similarity('赵子龙', '赵云')}")
# 寻找与 '赵子龙' 和 '刘备' 最相似，但与 '赵云' 最不相似的词 (新模型)
print(f"my_model.wv.most_similar(positive=['赵子龙', '刘备'], negative=['赵云']) => {my_model.wv.most_similar(positive=['赵子龙', '刘备'], negative=['赵云'])}")
