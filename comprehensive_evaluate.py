import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from config import Config

def load_real_mnist_samples(num_samples=1000):
    """从本地加载真实MNIST样本用于比较"""
    try:
        # 使用本地MNIST数据
        mnist_path = os.path.join(os.path.dirname(__file__), 'mnist_data', 'mnist.npz')
        with np.load(mnist_path) as data:
            x_train = data['x_train'].astype(np.float32) / 255.0
        # 随机选择样本
        indices = np.random.choice(len(x_train), num_samples, replace=False)
        real_samples = x_train[indices]
        return real_samples
    except Exception as e:
        print(f"无法加载本地MNIST数据: {e}")
        return None

def calculate_image_statistics(image_path):
    """计算单张图像的统计特征（不使用scipy）"""
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # 基本统计特征
    mean_intensity = np.mean(img_array)
    std_intensity = np.std(img_array)
    contrast = np.max(img_array) - np.min(img_array)
    
    # 简单的边缘检测（不使用scipy）
    # 使用简单的梯度计算
    h, w = img_array.shape
    grad_x = np.zeros_like(img_array)
    grad_y = np.zeros_like(img_array)
    
    # 计算x方向梯度
    grad_x[:, 1:] = img_array[:, 1:] - img_array[:, :-1]
    # 计算y方向梯度
    grad_y[1:, :] = img_array[1:, :] - img_array[:-1, :]
    
    # 计算梯度幅值
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_density = np.mean(edge_magnitude)
    
    return {
        'mean': mean_intensity,
        'std': std_intensity,
        'contrast': contrast,
        'edge_density': edge_density
    }

def analyze_image_quality(image_files, images_dir):
    """分析最后一个epoch的图像质量"""
    print("=== 图像质量分析（最新epoch） ===")
    print("说明：")
    print("  - 平均亮度：值越高图像越亮，理想范围0.3-0.7")
    print("  - 标准差：值越高对比度越强，理想范围0.2-0.5") 
    print("  - 对比度：值越高图像越清晰，理想范围0.5-1.0")
    print("  - 边缘密度：值越高细节越丰富，理想范围0.1-0.3")
    print()
    
    if not image_files:
        print("没有找到图像文件")
        return
    
    # 只分析最后一个epoch的图像
    latest_file = image_files[-1]
    latest_img_path = os.path.join(images_dir, latest_file)
    epoch = int(latest_file.split('_')[1].split('.')[0])
    
    try:
        stats = calculate_image_statistics(latest_img_path)
        print(f"Epoch {epoch} 图像质量指标:")
        print(f"  平均亮度: {stats['mean']:.3f}")
        print(f"    解释: {'亮度适中' if 0.3 <= stats['mean'] <= 0.7 else '亮度偏' + ('高' if stats['mean'] > 0.7 else '低')}")
        print(f"  标准差: {stats['std']:.3f}")
        print(f"    解释: {'对比度适中' if 0.2 <= stats['std'] <= 0.5 else '对比度偏' + ('高' if stats['std'] > 0.5 else '低')}")
        print(f"  对比度: {stats['contrast']:.3f}")
        print(f"    解释: {'清晰度良好' if stats['contrast'] > 0.5 else '清晰度需要改进'}")
        print(f"  边缘密度: {stats['edge_density']:.3f}")
        print(f"    解释: {'细节丰富' if 0.1 <= stats['edge_density'] <= 0.3 else '细节' + ('过多' if stats['edge_density'] > 0.3 else '不足')}")
        
        # 显示最新图像
        img = mpimg.imread(latest_img_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f'Latest Generated Image (Epoch {epoch})')
        plt.axis('off')
        plt.savefig('Number_GAN/images/latest_generated.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"分析图像质量时出错: {e}")

def analyze_diversity(image_files, images_dir):
    """分析生成多样性（比较最后几个epoch）"""
    print("\n=== 多样性分析（最后5个epoch） ===")
    print("说明：")
    print("  - 相邻图像差异：值越高表示生成器越不稳定，值过低可能表示模式崩塌")
    print("  - 理想范围：0.02-0.15，过低说明生成器陷入局部最优")
    print("  - 过高说明训练不稳定，过低说明缺乏多样性")
    print()
    
    # 只分析最后5个epoch
    recent_files = image_files[-5:] if len(image_files) >= 5 else image_files
    
    diversity_scores = []
    epochs = []
    
    for i in range(len(recent_files) - 1):
        try:
            img1_path = os.path.join(images_dir, recent_files[i])
            img2_path = os.path.join(images_dir, recent_files[i + 1])
            
            img1 = np.array(Image.open(img1_path).convert('L')).astype(np.float32) / 255.0
            img2 = np.array(Image.open(img2_path).convert('L')).astype(np.float32) / 255.0
            
            # 计算图像差异
            diff = np.mean(np.abs(img1 - img2))
            diversity_scores.append(diff)
            epochs.append(int(recent_files[i].split('_')[1].split('.')[0]))
            
        except Exception as e:
            print(f"计算多样性时出错: {e}")
    
    if diversity_scores:
        avg_diversity = np.mean(diversity_scores)
        print(f"最近{len(recent_files)}个epoch的多样性指标:")
        print(f"  平均相邻差异: {avg_diversity:.4f}")
        print(f"    解释: {'训练稳定，生成质量良好' if 0.02 <= avg_diversity <= 0.15 else '可能存在问题'}")
        print(f"  多样性变化范围: {min(diversity_scores):.4f} - {max(diversity_scores):.4f}")
        print(f"    解释: 范围越大说明训练过程越不稳定")
        
        # 显示最后几个epoch的图像对比
        fig, axes = plt.subplots(1, len(recent_files), figsize=(15, 3))
        if len(recent_files) == 1:
            axes = [axes]
        
        for i, file in enumerate(recent_files):
            epoch = int(file.split('_')[1].split('.')[0])
            img_path = os.path.join(images_dir, file)
            img = mpimg.imread(img_path)
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Epoch {epoch}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('Number_GAN/images/recent_epochs_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

def compare_with_real_data(image_files, images_dir):
    """与真实数据比较（只比较最新epoch）"""
    print("\n=== 与真实数据比较（最新epoch） ===")
    print("说明：")
    print("  - 相似度分数：1.0表示完全相似，0.0表示完全不同")
    print("  - 均值相似度：反映整体亮度分布是否接近真实数据")
    print("  - 标准差相似度：反映对比度分布是否接近真实数据")
    print("  - 对比度相似度：反映图像清晰度是否接近真实数据")
    print("  - 总体相似度：综合评估生成质量，>0.7为良好")
    print()
    
    # 加载真实MNIST数据
    real_samples = load_real_mnist_samples(1000)
    
    if real_samples is None:
        print("无法加载真实数据进行比较")
        return
    
    # 分析最新生成的图像
    if image_files:
        latest_file = image_files[-1]
        latest_img_path = os.path.join(images_dir, latest_file)
        epoch = int(latest_file.split('_')[1].split('.')[0])
        
        try:
            # 加载最新生成的图像
            generated_img = np.array(Image.open(latest_img_path).convert('L')).astype(np.float32) / 255.0
            
            # 计算真实数据的统计特征
            real_mean = np.mean(real_samples)
            real_std = np.std(real_samples)
            real_contrast = np.mean([np.max(sample) - np.min(sample) for sample in real_samples])
            
            # 计算生成数据的统计特征
            gen_mean = np.mean(generated_img)
            gen_std = np.std(generated_img)
            gen_contrast = np.max(generated_img) - np.min(generated_img)
            
            print(f"Epoch {epoch} 统计特征比较:")
            print(f"  真实数据 - 均值: {real_mean:.3f}, 标准差: {real_std:.3f}, 对比度: {real_contrast:.3f}")
            print(f"  生成数据 - 均值: {gen_mean:.3f}, 标准差: {gen_std:.3f}, 对比度: {gen_contrast:.3f}")
            
            # 计算相似度分数
            mean_similarity = 1 - abs(real_mean - gen_mean)
            std_similarity = 1 - abs(real_std - gen_std)
            contrast_similarity = 1 - abs(real_contrast - gen_contrast)
            
            overall_similarity = (mean_similarity + std_similarity + contrast_similarity) / 3
            
            print(f"\n相似度分析:")
            print(f"  均值相似度: {mean_similarity:.3f}")
            print(f"    解释: {'亮度分布接近真实数据' if mean_similarity > 0.8 else '亮度分布与真实数据差异较大'}")
            print(f"  标准差相似度: {std_similarity:.3f}")
            print(f"    解释: {'对比度分布接近真实数据' if std_similarity > 0.8 else '对比度分布与真实数据差异较大'}")
            print(f"  对比度相似度: {contrast_similarity:.3f}")
            print(f"    解释: {'清晰度接近真实数据' if contrast_similarity > 0.8 else '清晰度与真实数据差异较大'}")
            print(f"  总体相似度: {overall_similarity:.3f}")
            print(f"    解释: {'生成质量优秀' if overall_similarity > 0.8 else '生成质量良好' if overall_similarity > 0.6 else '生成质量需要改进'}")
            
        except Exception as e:
            print(f"比较分析时出错: {e}")

def analyze_training_stability(image_files, images_dir):
    """分析训练稳定性（简化版）"""
    print("\n=== 训练稳定性分析（简化版） ===")
    print("说明：")
    print("  - 文件大小：反映生成图像的信息量，通常质量越高文件越大")
    print("  - 最后几个epoch的变化趋势：上升表示质量在改善")
    print("  - 理想情况：文件大小逐渐增加并趋于稳定")
    print()
    
    # 只分析最后10个epoch
    recent_files = image_files[-10:] if len(image_files) >= 10 else image_files
    
    file_sizes = []
    epochs = []
    
    for file in recent_files:
        try:
            epoch = int(file.split('_')[1].split('.')[0])
            file_size = os.path.getsize(os.path.join(images_dir, file))
            file_sizes.append(file_size)
            epochs.append(epoch)
        except:
            continue
    
    if file_sizes:
        # 计算稳定性指标
        size_variance = np.var(file_sizes)
        size_trend = np.polyfit(epochs, file_sizes, 1)[0]  # 线性趋势斜率
        
        print(f"最近{len(recent_files)}个epoch的稳定性指标:")
        print(f"  文件大小方差: {size_variance:.0f}")
        print(f"    解释: {'训练非常稳定' if size_variance < 10000 else '训练比较稳定' if size_variance < 50000 else '训练不够稳定'}")
        print(f"  大小变化趋势: {'上升' if size_trend > 0 else '下降' if size_trend < 0 else '稳定'}")
        print(f"    解释: {'质量在改善' if size_trend > 0 else '可能过拟合' if size_trend < 0 else '训练趋于稳定'}")
        print(f"  平均文件大小: {np.mean(file_sizes)/1024:.1f} KB")
        print(f"    解释: {'文件大小适中' if 50 < np.mean(file_sizes)/1024 < 200 else '文件过大或过小'}")
        
        # 绘制最近epoch的稳定性图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, file_sizes)
        plt.title('Recent Training Stability - File Size Change')
        plt.xlabel('Epoch')
        plt.ylabel('File Size (bytes)')
        plt.grid(True)
        plt.savefig('Number_GAN/images/recent_stability_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """主评估函数"""
    print("=== GAN生成结果评估（最新epoch） ===\n")
    print("本评估专注于最新epoch的生成质量：")
    print("1. 图像质量分析 - 评估最新生成图像的清晰度、对比度等")
    print("2. 多样性分析 - 评估最近几个epoch的稳定性和多样性")
    print("3. 与真实数据比较 - 评估最新生成质量与真实MNIST的相似度")
    print("4. 训练稳定性分析 - 评估最近训练过程的稳定性")
    print()
    
    config = Config()
    images_dir = config.OUTPUT_PATH
    
    if not os.path.exists(images_dir):
        print("未找到图像目录")
        return
    
    # 获取所有生成的图像文件
    image_files = [f for f in os.listdir(images_dir) if f.startswith('mnist_') and f.endswith('.png')]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(f"找到 {len(image_files)} 个训练检查点")
    
    if len(image_files) == 0:
        print("没有找到生成的图像文件")
        return
    
    latest_epoch = int(image_files[-1].split('_')[1].split('.')[0])
    print(f"最新epoch: {latest_epoch}")
    print()
    
    # 1. 图像质量分析
    analyze_image_quality(image_files, images_dir)
    
    # 2. 多样性分析
    analyze_diversity(image_files, images_dir)
    
    # 3. 与真实数据比较
    compare_with_real_data(image_files, images_dir)
    
    # 4. 训练稳定性分析
    analyze_training_stability(image_files, images_dir)
    
    print("\n=== 评估完成 ===")
    print("所有分析图表已保存到 Number_GAN/images/ 目录")
    print("\n评估总结：")
    print("- 如果所有指标都在理想范围内，说明GAN训练成功")
    print("- 如果某些指标异常，可能需要调整训练参数或网络结构")
    print("- 建议结合视觉观察来综合判断生成质量")

if __name__ == "__main__":
    main() 