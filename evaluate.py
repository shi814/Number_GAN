import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import os
from config import Config
from utils.data_loader import generate_noise

def load_trained_generator(model_path):
    """加载训练好的生成器"""
    try:
        # 这里需要根据你的模型保存方式调整
        generator = load_model(model_path)
        return generator
    except:
        print("无法加载模型，请检查模型路径")
        return None

def visual_evaluation(generator, num_samples=100):
    """视觉质量评估"""
    config = Config()
    
    # 生成样本
    noise = generate_noise(num_samples, config.LATENT_DIM)
    generated_images = generator.predict(noise)
    
    # 转换到[0,1]范围用于显示
    generated_images = 0.5 * generated_images + 0.5
    
    # 显示生成的图像
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            if idx < num_samples:
                axes[i, j].imshow(generated_images[idx, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
    
    plt.suptitle('Generated MNIST Digits', fontsize=16)
    plt.tight_layout()
    plt.savefig('Number_GAN/images/evaluation_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return generated_images

def diversity_analysis(generated_images):
    """多样性分析"""
    # 计算生成图像的方差（多样性指标）
    variance = np.var(generated_images)
    print(f"生成图像方差: {variance:.4f}")
    
    # 计算图像间的平均差异
    differences = []
    for i in range(len(generated_images)):
        for j in range(i+1, len(generated_images)):
            diff = np.mean(np.abs(generated_images[i] - generated_images[j]))
            differences.append(diff)
    
    avg_difference = np.mean(differences)
    print(f"图像间平均差异: {avg_difference:.4f}")
    
    return variance, avg_difference

def compare_with_real_data(generated_images):
    """与真实数据比较"""
    # 加载真实MNIST数据
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0  # 归一化到[0,1]
    x_train = np.expand_dims(x_train, axis=-1)
    
    # 随机选择真实样本进行比较
    real_samples = x_train[np.random.choice(len(x_train), len(generated_images), replace=False)]
    
    # 计算真实数据的统计信息
    real_mean = np.mean(real_samples)
    real_std = np.std(real_samples)
    
    # 计算生成数据的统计信息
    gen_mean = np.mean(generated_images)
    gen_std = np.std(generated_images)
    
    print(f"真实数据 - 均值: {real_mean:.4f}, 标准差: {real_std:.4f}")
    print(f"生成数据 - 均值: {gen_mean:.4f}, 标准差: {gen_std:.4f}")
    
    # 计算分布相似度
    distribution_similarity = 1 - abs(real_mean - gen_mean) - abs(real_std - gen_std)
    print(f"分布相似度: {distribution_similarity:.4f}")
    
    return distribution_similarity

def training_progress_analysis():
    """分析训练进度"""
    config = Config()
    images_dir = config.OUTPUT_PATH
    
    if not os.path.exists(images_dir):
        print("未找到生成的图像目录")
        return
    
    # 获取所有生成的图像文件
    image_files = [f for f in os.listdir(images_dir) if f.startswith('mnist_') and f.endswith('.png')]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(f"找到 {len(image_files)} 个训练检查点")
    print("训练进度分析:")
    for i, file in enumerate(image_files[::5]):  # 每5个文件显示一个
        epoch = int(file.split('_')[1].split('.')[0])
        print(f"  Epoch {epoch}: {file}")
    
    return image_files

def main():
    """主评估函数"""
    print("=== GAN生成结果评估 ===\n")
    
    # 1. 训练进度分析
    print("1. 训练进度分析:")
    image_files = training_progress_analysis()
    print()
    
    # 2. 加载最新模型（如果有的话）
    config = Config()
    latest_generator_path = f"{config.MODEL_PATH}/generator_latest.h5"
    
    if os.path.exists(latest_generator_path):
        print("2. 加载训练好的生成器...")
        generator = load_trained_generator(latest_generator_path)
        
        if generator is not None:
            # 3. 视觉评估
            print("3. 视觉质量评估...")
            generated_images = visual_evaluation(generator)
            
            # 4. 多样性分析
            print("4. 多样性分析...")
            diversity_analysis(generated_images)
            
            # 5. 与真实数据比较
            print("5. 与真实数据比较...")
            compare_with_real_data(generated_images)
        else:
            print("无法加载生成器模型")
    else:
        print("未找到训练好的模型文件")
        print("请先完成训练或检查模型保存路径")

if __name__ == "__main__":
    main() 