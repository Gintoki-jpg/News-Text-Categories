import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):
    def __init__(self):
        self.model_name = 'TextRNN'
        self.train_path = 'data/train.txt'                                # 训练集
        self.dev_path = 'data/dev.txt'                                    # 验证集
        self.test_path = 'data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open('data/class.txt').readlines()]  # 类别名单
        self.save_path = 'model/' + self.model_name + '.ckpt'               # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # 设备
        self.vocab_path = 'data/vocab.pkl'                                # 词表
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(np.load('data/embedding_SougouNews.npz')['embeddings'].astype('float32'))

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 300                                                # 字向量维度
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
    def forward(self, x):
        out = self.embedding(x[0])  # x[0]是句子
        out, _ = self.lstm(out)
        out = torch.mean(out, 1)
        out = self.fc(out)
        return out