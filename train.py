import numpy as np
from tqdm import tqdm  # 进度条工具
from models.gan import GAN  # GAN模型
from utils.data_loader import load_and_preprocess_data, sample_real_data, generate_noise
from utils.image_utils import save_generated_images, save_model_weights, create_output_dir
from config import Config

def train_gan():
    # 初始化配置
    config = Config()
    # 创建GAN实例（包含生成器、判别器、组合模型）
    gan = GAN()
    # 加载并预处理数据
    dataset = load_and_preprocess_data()
    # 创建输出目录
    create_output_dir()
    
    # 训练循环（使用tqdm显示进度条）
    for epoch in tqdm(range(config.EPOCHS), desc="Training GAN"):
        # ===== 1. 训练判别器 =====
        # 采样真实图像
        real_imgs = sample_real_data(dataset, config.BATCH_SIZE)
        # 生成假图像
        noise = generate_noise(config.BATCH_SIZE, config.LATENT_DIM)
        fake_imgs = gan.generator.predict(noise)
        
        # 标签平滑（真实标签=0.9，假标签=0.0）
        valid_labels = np.ones((config.BATCH_SIZE, 1)) * 0.9
        fake_labels = np.zeros((config.BATCH_SIZE, 1))
        
        # 用真实图像训练判别器
        d_loss_real = gan.discriminator.train_on_batch(real_imgs, valid_labels)
        # 用生成图像训练判别器
        d_loss_fake = gan.discriminator.train_on_batch(fake_imgs, fake_labels)
        # 计算平均损失
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ===== 2. 训练生成器 =====
        noise = generate_noise(config.BATCH_SIZE, config.LATENT_DIM)
        # 训练组合模型（目标：让判别器输出1）
        g_loss = gan.combined.train_on_batch(noise, np.ones((config.BATCH_SIZE, 1)))
        
        # 每100轮打印进度
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
        
        # 定期保存结果
        if epoch % config.SAVE_INTERVAL == 0:
            # 保存生成的图像
            save_generated_images(gan.generator, epoch)
            # 保存模型权重
            save_model_weights(gan.generator, gan.discriminator, epoch)
    
    return gan  # 返回训练好的GAN