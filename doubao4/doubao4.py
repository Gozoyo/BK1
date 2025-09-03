import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_scratches_and_folds(image_path):
    """
    优化后的老照片划痕和折痕检测函数
    减少对正常纹理的误检，增强真实损伤的识别
    """
    # 读取图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    
    # 转换为灰度图
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # 增强去噪处理（针对老照片特点优化）
    # 先使用高斯模糊初步去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 再使用非局部均值去噪，增强对颗粒噪声的去除
    denoised = cv2.fastNlMeansDenoising(blurred, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # 改进的边缘检测：提高阈值减少误检
    # 高阈值提高到180，低阈值相应提高到80，减少对弱边缘的检测
    edges = cv2.Canny(denoised, threshold1=80, threshold2=180)
    
    # 形态学操作优化：先腐蚀后膨胀，去除细小噪声边缘
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)  # 先腐蚀去除小噪声
    edges = cv2.dilate(edges, kernel, iterations=1)  # 再膨胀恢复有效边缘
    
    # 改进的霍夫直线检测：只检测足够长的连续线条
    lines = cv2.HoughLinesP(
        edges,
        rho=1, 
        theta=np.pi/180,
        threshold=40,       # 提高阈值，减少候选直线
        minLineLength=40,   # 只检测长度≥40的线段（过滤短划痕噪声）
        maxLineGap=15       # 允许线段间有15的间隙（适应断续的折痕）
    )
    
    # 创建标记图像
    detected_image = original.copy()
    scratch_count = 0
    
    # 标记检测到的线条（划痕和折痕）
    if lines is not None:
        scratch_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算线段长度，过滤极短线条
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)** 2)
            if length > 30:  # 再次过滤短线条
                cv2.line(detected_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return original, detected_image, edges, scratch_count

def main():
    image_path = "image2.png"  # 替换为你的图像路径
    
    try:
        original, detected_image, edges, count = detect_scratches_and_folds(image_path)
        
        # 保存结果
        output_path = "improved_detected_scratches.png"
        cv2.imwrite(output_path, detected_image)
        print(f"检测完成，共发现 {count} 处可能的划痕或折痕")
        print(f"结果已保存到 {output_path}")
        
        # 显示结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis("off")
        
        plt.subplot(132)
        plt.imshow(edges, cmap="gray")
        plt.title("优化后的边缘检测")
        plt.axis("off")
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        plt.title(f"检测结果（{count}处损伤）")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"处理出错: {str(e)}")

if __name__ == "__main__":
    main()
    