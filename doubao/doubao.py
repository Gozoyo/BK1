import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_bad_pixels(image_path, threshold=30, window_size=3):
    """
    检测图像中的坏点
    
    参数:
    - image_path: 图像路径
    - threshold: 判断为坏点的阈值，值越大检测越严格
    - window_size: 计算周围像素平均值的窗口大小
    
    返回:
    - original: 原始图像
    - bad_pixels: 坏点坐标列表
    - marked_image: 标记了坏点的图像
    """
    # 读取图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    
    # 转换为灰度图以便处理
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 创建一个与原图相同大小的数组用于标记坏点
    bad_pixels = []
    
    # 计算窗口半径
    radius = window_size // 2
    
    # 遍历图像中的每个像素
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            # 获取当前像素值
            current_pixel = gray[y, x]
            
            # 获取周围像素值（排除中心像素）
            window = gray[y-radius:y+radius+1, x-radius:x+radius+1]
            window_flat = window.flatten()
            surrounding_pixels = np.delete(window_flat, window_flat.size // 2)
            
            # 计算周围像素的平均值和标准差
            mean = np.mean(surrounding_pixels)
            std = np.std(surrounding_pixels)
            
            # 判断当前像素是否为坏点
            # 如果与平均值的差异超过阈值倍的标准差，则视为坏点
            if abs(current_pixel - mean) > threshold * std:
                bad_pixels.append((x, y))
    
    # 创建标记了坏点的图像
    marked_image = original.copy()
    for (x, y) in bad_pixels:
        # 用红色标记坏点
        cv2.circle(marked_image, (x, y), 2, (0, 0, 255), -1)
    
    return original, bad_pixels, marked_image

def main():
    # 图像路径
    image_path = "image2.png"
    
    # 检测坏点
    try:
        original, bad_pixels, marked_image = detect_bad_pixels(
            image_path, 
            threshold=2.5,  # 可根据实际情况调整
            window_size=3   # 3x3窗口
        )
        
        # 输出检测结果
        print(f"检测到 {len(bad_pixels)} 个坏点")
        
        # 保存标记了坏点的图像
        output_path = "image2_with_bad_pixels.png"
        cv2.imwrite(output_path, marked_image)
        print(f"已保存标记坏点的图像到 {output_path}")
        
        # 显示结果（需要图形界面支持）
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis("off")
        
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        plt.title(f"标记了坏点的图像 (共{len(bad_pixels)}个)")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
