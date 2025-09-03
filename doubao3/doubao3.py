import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_scratches_and_folds(image_path, threshold1=50, threshold2=150, min_line_length=30, max_line_gap=10):
    """
    检测老照片中的划痕和折痕
    
    参数:
    - image_path: 图像路径
    - threshold1: Canny边缘检测低阈值
    - threshold2: Canny边缘检测高阈值
    - min_line_length: 检测线段的最小长度
    - max_line_gap: 线段之间允许的最大间隙
    
    返回:
    - original: 原始图像
    - detected_image: 标记了划痕和折痕的图像
    - edges: 边缘检测结果
    """
    # 读取图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    
    # 转换为灰度图
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # 去噪处理（减少干扰）
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 边缘检测（Canny算法对线性特征敏感）
    edges = cv2.Canny(denoised, threshold1, threshold2)
    
    # 形态学操作：增强线条特征
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)  # 增强边缘
    edges = cv2.erode(edges, kernel, iterations=1)   # 细化边缘
    
    # 使用霍夫变换检测直线（适合检测划痕和折痕）
    lines = cv2.HoughLinesP(
        edges,
        rho=1,               # 像素精度
        theta=np.pi/180,     # 角度精度
        threshold=30,        # 直线判定阈值
        minLineLength=min_line_length,  # 最小线段长度
        maxLineGap=max_line_gap         # 线段最大间隙
    )
    
    # 创建标记图像
    detected_image = original.copy()
    
    # 标记检测到的线条（划痕和折痕）
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 用红色标记检测到的线条
            cv2.line(detected_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return original, detected_image, edges

def main():
    # 图像路径
    image_path = "image2.png"
    
    try:
        # 检测划痕和折痕，可根据实际情况调整参数
        original, detected_image, edges = detect_scratches_and_folds(
            image_path,
            threshold1=30,       # 降低低阈值可检测更多弱边缘
            threshold2=100,      # 高阈值控制强边缘
            min_line_length=20,  # 较小值可检测短划痕
            max_line_gap=5       # 较小值可检测连续线条
        )
        
        # 保存结果
        output_path = "detected_scratches.png"
        cv2.imwrite(output_path, detected_image)
        print(f"已保存标记结果到 {output_path}")
        
        # 显示结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis("off")
        
        plt.subplot(132)
        plt.imshow(edges, cmap="gray")
        plt.title("边缘检测结果")
        plt.axis("off")
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        plt.title("标记的划痕和折痕")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()