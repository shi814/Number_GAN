from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from config import Config
from .generator import build_generator
from .discriminator import build_discriminator

class GAN:
    def __init__(self):
        self.config = Config()
        #创建adam优化器（带自定义学习率和动量）
        optimizer = Adam(self.config.LEARNING_RATE,self.config.BETA_1)

        #构建并编译判别器
        self.discriminator = build_discriminator()
        self.discriminator.compile(
            loss ='binary_crossentropy',#二元交叉熵损失
            optimizer = optimizer,#使用上面定义的优化器
            metrics = ['accuracy'])#跟踪准确率
        
        #构建生成器
        self.generator = build_generator()

        #构建组合模型（生成器+冻结的判别器）
        z = Input(shape=(self.config.LATENT_DIM,)) #噪声输入
        img = self.generator(z) #生成图像

        self.discriminator.trainable = False #关键：冻结判别器权重
        valid = self.discriminator(img) #判别结果

        #创建端到端的GAN模型（噪声到判别结果）
        self.combined = Model(z,valid)
        #编译组合模型（只训练生成器）
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)