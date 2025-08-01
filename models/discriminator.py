from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten
from config import Config

def build_discriminator():
    "构建判别器模型"
    config = Config()
    model = Sequential()
    #输入层（28*28*1到1024）
    model.add(Flatten(input_shape=config.IMG_SHAPE)) #展平

    #中间层（1024到512到256）
    for units in config.DISCRIMINATOR_DENSE_UNITS:
        model.add(Dense(units)) #全连接层
        model.add(LeakyReLU(alpha=config.LEAKY_RELU_ALPHA)) #Leaky ReLU激活函数
    
    #输出层（256到1）
    model.add(Dense(1,activation='sigmoid')) #sigmoid激活函数,二分类输出

    return model #返回判别器