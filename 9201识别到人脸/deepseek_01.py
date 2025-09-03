import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def protect_facial_features(image):
    """
    创建面部特征保护掩膜 - 使用绝对路径
    """
    # 获取OpenCV安装路径
    cv2_path = os.path.dirname(cv2.__file__)
    data_path = os.path.join(cv2_path, 'data')
    
    # 构建完整的XML文件路径
    face_cascade_path = os.path.join(data_path, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(data_path, 'haarcascade_eye.xml')
    
    # 检查文件是否存在
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"找不到人脸检测文件: {face_cascade_path}")
    if not os.path.exists(eye_cascade_path):
        raise FileNotFoundError(f"找不到眼睛检测文件: {eye_cascade_path}")
    
    # 初始化检测器
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 创建全白掩膜
    protection_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # 扩展保护区域，确保覆盖整个面部
        cv2.rectangle(protection_mask, (max(0, x-10), max(0, y-10)), 
                     (min(width, x+w+10), min(height, y+h+10)), 0, -1)
    
    return protection_mask

def detect_scratches_folds_safe(image_path, sensitivity=50):
    """
    安全的划痕检测函数，不依赖人脸检测
    """
    # 读取图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("无法读取图像，请检查路径")
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 1. 手动创建保护区域（避免使用人脸检测）
    protection_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 手动指定可能的面部区域（您可以根据图像调整这些值）
    # 通常面部在图像中心区域
    center_x, center_y = width // 2, height // 2
    face_size = min(width, height) // 3
    
    # 在图像中心创建圆形保护区域
    cv2.circle(protection_mask, (center_x, center_y), face_size, 0, -1)
    
    # 2. 应用保护掩膜
    protected_gray = gray.copy()
    protected_gray[protection_mask == 0] = np.mean(gray)
    
    # 3. 边缘检测（使用更保守的参数）
    blurred = cv2.GaussianBlur(protected_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)  # 提高阈值，减少误检
    
    # 移除保护区域的边缘
    edges[protection_mask == 0] = 0
    
    # 4. 只保留较长的边缘（划痕通常较长）
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 5. 霍夫直线检测（更严格的条件）
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=25,  # 提高阈值
        minLineLength=50,  # 增加最小长度
        maxLineGap=10     # 减少最大间隙
    )
    
    # 创建修复掩膜
    repair_mask = np.zeros_like(gray)
    detected = original.copy()
    count = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 确保线条不在保护区域内
            line_mask = np.zeros_like(gray)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            if np.sum((line_mask > 0) & (protection_mask == 0)) == 0:
                cv2.line(detected, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(repair_mask, (x1, y1), (x2, y2), 255, 3)
                count += 1
    
    return original, edges, detected, repair_mask, count, protection_mask

def main_safe():
    image_path = "image3.jpg"
    sensitivity = 50  # 使用更保守的敏感度
    
    try:
        original, edges, detected, repair_mask, count, protection_mask = detect_scratches_folds_safe(
            image_path, sensitivity=sensitivity
        )
        
        print(f"检测到 {count} 处可能的划痕/折痕")
        
        # 修复图像
        if count > 0:
            repaired_img = cv2.inpaint(original, repair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        else:
            repaired_img = original.copy()
            print("未检测到需要修复的划痕")
        
        # 显示结果
        plt.figure(figsize=(18, 6))
        
        plt.subplot(231), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原图"), plt.axis('off')
        
        plt.subplot(232), plt.imshow(protection_mask, cmap="gray")
        plt.title("保护区域\n(黑色=不检测区域)"), plt.axis('off')
        
        plt.subplot(233), plt.imshow(edges, cmap="gray")
        plt.title("边缘检测结果"), plt.axis('off')
        
        plt.subplot(234), plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        plt.title(f"检测到的划痕\n({count}处)"), plt.axis('off')
        
        plt.subplot(235), plt.imshow(repair_mask, cmap="gray")
        plt.title("修复掩膜"), plt.axis('off')
        
        plt.subplot(236), plt.imshow(cv2.cvtColor(repaired_img, cv2.COLOR_BGR2RGB))
        plt.title("修复后图像"), plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 保存结果
        cv2.imwrite("safe_detected.png", detected)
        cv2.imwrite("safe_protection_mask.png", protection_mask * 255)
        cv2.imwrite("safe_repair_mask.png", repair_mask)
        cv2.imwrite("safe_repaired_image.png", repaired_img)
        
    except Exception as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    main_safe()