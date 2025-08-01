import os
from turtle import filling
import matplotlib.pyplot as plt 
from config import Config
from utils.data_loader import generate_noise

def create_output_dir():
    """创建输出目录"""
    config = Config()
    #递归创建目录（exist_ok=True表示如果目录存在，不报错）
    os.makedirs(config.OUTPUT_DIR,exist_ok = Ture)
    os.makedirs(config.MODEL_DIR,exist_ok = Ture)

def save_generated_images(generator,epoch,rows=5,cols=5):
    """保存生成的图像网络"""
    config = Config()
    # 生成噪声向量
    noise = generate_noise(rows * cols,config.LATENT_DIM):
    #用生成器生成图像
    gen_imgs = generator.predict(noise)

    #从[-1,1]转换到[0,1](matplotlib要求)
    gen_imgs = 0.5 * gen_imgs + 0.5

    #创建子图网格
    fig,axs = plt.subplots(rows,cols)
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            # 显示灰度图像
            axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
            axs[i,j].axis('off')#关闭坐标轴
            cnt += 1
    #  保存图像文件
    fig.savefig(f"{config.OUTPUT_DIR}/mnist_{epoch}.png")
    plt.close()

def save_model_weights(generator,discriminator,epoch):
    """保存模型权重"""
    config = Config()
    #生成器
    generator.save_weights(f"{config.MODEL_DIR}/generator_{epoch}.h5")
    #判别器
    discriminator.save_weights(f"{config.MODEL_DIR}/discriminator_{epoch}.h5")


