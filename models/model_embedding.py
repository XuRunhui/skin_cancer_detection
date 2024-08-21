import torch
import torch.nn as nn

# 定义词汇表大小（vocab_size）和嵌入维度（embedding_dim）
vocab_size = 10  # 词汇表大小
embedding_dim = 3  # 嵌入维度

# 创建嵌入层
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 创建一个包含索引的示例输入（这里每个数字代表一个单词在词汇表中的索引）
input_indices = torch.tensor([1, 3, 4, 7])  # 输入的索引序列

# 获取嵌入向量
embedded_output = embedding(input_indices)

# 输出嵌入向量
print("嵌入向量：\n", embedded_output)


