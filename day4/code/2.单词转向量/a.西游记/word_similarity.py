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

# 打印 '孙悟空' 的词向量
print(f"model.wv['孙悟空'] => {model.wv['孙悟空']}")
# 计算 '孙悟空' 和 '猪八戒' 的相似度
print(f"model.wv.similarity('孙悟空', '猪八戒') => {model.wv.similarity('孙悟空', '猪八戒')}")
# 计算 '孙悟空' 和 '孙行者' 的相似度
print(f"model.wv.similarity('孙悟空', '孙行者') => {model.wv.similarity('孙悟空', '孙行者')}")
# 寻找与 '孙悟空' 和 '唐僧' 最相似，但与 '孙行者' 最不相似的词
print(f"model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']) => {model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者'])}")

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
my_model.save('./models/西游记相似度匹配.model')

# 计算 '孙悟空' 和 '猪八戒' 的相似度 (新模型)
print(f"my_model.wv.similarity('孙悟空', '猪八戒') => {my_model.wv.similarity('孙悟空', '猪八戒')}")
# 计算 '孙悟空' 和 '孙行者' 的相似度 (新模型)
print(f"my_model.wv.similarity('孙悟空', '孙行者') => {my_model.wv.similarity('孙悟空', '孙行者')}")
# 寻找与 '孙悟空' 和 '唐僧' 最相似，但与 '孙行者' 最不相似的词 (新模型)
print(f"my_model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']) => {my_model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者'])}")
