#配置文件
class Config:
    # 数据集配置(MNIST,28*28,1灰度图)
    IMG_ROWS = 28 #高
    IMG_COLS = 28 #宽
    CHANNELS = 1 #通道数
    IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS) #图像形状
    LATENT_DIM = 100 #噪声向量维度
    # 训练参数
    EPOCHS = 30000 #训练轮数
    BATCH_SIZE = 32 #每批样本数量
    SAVE_INTERVAL = 200 #每200轮保存一次模型
    # 优化器参数
    LEARNING_RATE = 0.0002 #ADAM优化器学习率
    BETA_1 = 0.5 #ADAM优化器第一动量参数
    BETA_2 = 0.999 #ADAM优化第二器动量参数 
    # 路径配置
    OUTPUT_PATH = "Number_GAN/images" #生成图片保存路径
    MODEL_PATH = "Number_GAN/models" #模型保存路径
    #网络参数
    GENERATOR_DENSE_UNITS = [256,512,1024] #生成器各层神经元数
    DISCRIMINATOR_DENSE_UNITS = [256,512,1024] #判别器各层神经元数
    LEAKY_RELU_ALPHA = 0.2 #Leaky ReLU激活函数负斜率系数
    BATCH_NORM_ALPH = 0.8 #批归一化参数
