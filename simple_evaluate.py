import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import Config

def analyze_generated_images():
    """分析已生成的图像"""
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
    
    # 显示训练进度
    print("\n=== 训练进度分析 ===")
    for i, file in enumerate(image_files[::5]):  # 每5个文件显示一个
        epoch = int(file.split('_')[1].split('.')[0])
        file_size = os.path.getsize(os.path.join(images_dir, file))
        print(f"  Epoch {epoch}: {file} ({file_size/1024:.1f} KB)")
    
    # 显示最新的几个图像
    print(f"\n=== 最新生成的图像 ===")
    latest_files = image_files[-3:]  # 显示最新的3个文件
    
    fig, axes = plt.subplots(1, len(latest_files), figsize=(15, 5))
    if len(latest_files) == 1:
        axes = [axes]
    
    for i, file in enumerate(latest_files):
        epoch = int(file.split('_')[1].split('.')[0])
        img_path = os.path.join(images_dir, file)
        img = mpimg.imread(img_path)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Epoch {epoch}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('Number_GAN/images/training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 分析文件大小变化（质量指标）
    print(f"\n=== 图像质量分析 ===")
    file_sizes = []
    epochs = []
    
    for file in image_files:
        epoch = int(file.split('_')[1].split('.')[0])
        file_size = os.path.getsize(os.path.join(images_dir, file))
        file_sizes.append(file_size)
        epochs.append(epoch)
    
    print(f"文件大小范围: {min(file_sizes)/1024:.1f} KB - {max(file_sizes)/1024:.1f} KB")
    print(f"平均文件大小: {sum(file_sizes)/len(file_sizes)/1024:.1f} KB")
    
    # 绘制文件大小变化图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, file_sizes)
    plt.xlabel('Epoch')
    plt.ylabel('File Size (bytes)')
    plt.title('Generated Image Quality Over Training')
    plt.grid(True)
    plt.savefig('Number_GAN/images/quality_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return image_files

def main():
    """主函数"""
    print("=== GAN生成结果简单评估 ===\n")
    analyze_generated_images()

if __name__ == "__main__":
    main() 