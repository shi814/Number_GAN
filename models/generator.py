from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape
from config import Config

def build_generator():
    "构建生成器模型"
    config = Config()
    model = Sequential()
    # 输入层（100维随机噪声转为256维）
    model.add(Dense(config.GENERATOR_DENSE_UNITS[0], input_dim=config.LATENT_DIM)) #全连接层
    model.add(LeakyReLU(alpha=config.LEAKY_RELU_ALPHA)) #Leaky ReLU激活函数，防止梯度消失
    model.add(BatchNormalization(momentum=config.BATCH_NORM_ALPH)) #批归一化

    #中间层（256到512到1024）
    for units in config.GENERATOR_DENSE_UNITS[1:]:
        model.add(Dense(units)) #全连接层
        model.add(LeakyReLU(alpha=config.LEAKY_RELU_ALPHA)) #Leaky ReLU激活函数
        model.add(BatchNormalization(momentum=config.BATCH_NORM_ALPH)) #批归一化
    
    #输出层（1024到28*28*1）
    model.add(Dense(config.IMG_ROWS*config.IMG_COLS*config.CHANNELS,activation='tanh')) #28*28*1,tanh激活函数，输出范围[-1,1]
    model.add(Reshape(config.IMG_SHAPE)) #重塑为28*28*1

    return model #返回生成器
