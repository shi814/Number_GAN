import numpy as np 
from tensorflow.keras.datasets import mnist 
from config import Config

def load_and_preprocess_data():
    """加载并预处理mnist数据集"""
    config = Config()
    #加载数据集（忽略标签，无监督学习），一次性读取60000张手写数字图片，x_train是60000*28*28的矩阵，_是标签
    (x_train, _), (_, _) = mnist.load_data()

    #归一化：原始像素0～255，归一化到[-1,1]
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    #添加通道维度：（60000，28，28）->(60000,28,28,1)，卷积层需要通道维
    x_train = np.expand_dims(x_train, axis=-1)
    
    return x_train #返回处理好的数据组，后面训练循环直接用它作为真图

def sample_real_data(dataset,batch_size):
    """从真实数据中采样一个批次"""
    #随机选择batch_size张图片
    idx = np.random.randint(0,dataset.shape[0],batch_size)
    return dataset[idx]#返回对应图像批次

def generate_noise(batch_size,latent_dim):
    """生成随机噪声向量"""
    #从标准正态分布采样噪声
    return np.random.normal(0,1,(batch_size,latent_dim))
