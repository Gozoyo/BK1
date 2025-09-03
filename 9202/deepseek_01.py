import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_scratches_folds_enhanced(image_path, sensitivity=70):  # 提高默认敏感度
    """
    增强的划痕检测函数，增加修复力度
    """
    # 读取图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("无法读取图像，请检查路径")
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 1. 手动创建保护区域
    protection_mask = np.ones((height, width), dtype=np.uint8) * 255
    center_x, center_y = width // 2, height // 2
    face_size = min(width, height) // 3
    cv2.circle(protection_mask, (center_x, center_y), face_size, 0, -1)
    
    # 2. 应用保护掩膜
    protected_gray = gray.copy()
    protected_gray[protection_mask == 0] = np.mean(gray)
    
    # 3. 增强的边缘检测 - 降低阈值，检测更多边缘
    blurred = cv2.GaussianBlur(protected_gray, (3, 3), 0)  # 减小模糊核
    edges = cv2.Canny(blurred, 20, 60)  # 大幅降低阈值！原来(30,100)
    
    # 移除保护区域的边缘
    edges[protection_mask == 0] = 0
    
    # 4. 增强的形态学操作 - 更好地连接断开的划痕
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)  # 增加膨胀次数
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 放宽霍夫直线检测条件 - 检测更多划痕
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=15,  # 降低阈值！原来25
        minLineLength=30,  # 减少最小长度！原来50
        maxLineGap=15     # 增加最大间隙！原来10
    )
    
    # 创建修复掩膜 - 增加线宽以确保完全覆盖划痕
    repair_mask = np.zeros_like(gray)
    detected = original.copy()
    count = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_mask = np.zeros_like(gray)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            if np.sum((line_mask > 0) & (protection_mask == 0)) == 0:
                cv2.line(detected, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 增加修复线宽！原来3，现在5
                cv2.line(repair_mask, (x1, y1), (x2, y2), 255, 5)
                count += 1
    
    # 6. 额外的划痕增强：检测可能被遗漏的短划痕
    # 查找轮廓来补充直线检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 只处理较长的轮廓
        if cv2.arcLength(contour, True) > 40:  # 长度阈值
            # 获取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            # 检查是否在保护区域内
            if protection_mask[y:y+h, x:x+w].mean() > 200:  # 不在保护区域
                # 在修复掩膜上绘制轮廓
                cv2.drawContours(repair_mask, [contour], -1, 255, 3)
                cv2.drawContours(detected, [contour], -1, (0, 255, 0), 2)  # 用绿色标记
    
    return original, edges, detected, repair_mask, count, protection_mask, height, width

def enhanced_inpaint(original, repair_mask):
    """
    增强的修复函数，使用更强的修复参数
    """
    # 方法1：使用更大的修复半径
    repaired_img = cv2.inpaint(original, repair_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # 方法2：多次修复（可选，对于顽固划痕）
    if np.sum(repair_mask > 0) > 1000:  # 如果划痕区域较大
        repaired_img = cv2.inpaint(repaired_img, repair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return repaired_img

def main_enhanced():
    image_path = "image3.jpg"
    sensitivity = 70  # 使用中等敏感度
    
    try:
        print("正在进行增强的划痕检测...")
        original, edges, detected, repair_mask, count, protection_mask, height, width = detect_scratches_folds_enhanced(
            image_path, sensitivity=sensitivity
        )
        
        print(f"检测到 {count} 处可能的划痕/折痕")
        
        # 增强修复
        if count > 0:
            print("正在进行增强修复...")
            repaired_img = enhanced_inpaint(original, repair_mask)
        else:
            repaired_img = original.copy()
            print("未检测到需要修复的划痕")
        
        # 显示对比结果
        plt.figure(figsize=(20, 8))
        
        plt.subplot(241), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原图"), plt.axis('off')
        
        plt.subplot(242), plt.imshow(protection_mask, cmap="gray")
        plt.title("保护区域"), plt.axis('off')
        
        plt.subplot(243), plt.imshow(edges, cmap="gray")
        plt.title("边缘检测"), plt.axis('off')
        
        plt.subplot(244), plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        plt.title(f"检测结果\n({count}处)"), plt.axis('off')
        
        plt.subplot(245), plt.imshow(repair_mask, cmap="gray")
        plt.title("修复掩膜"), plt.axis('off')
        
        plt.subplot(246), plt.imshow(cv2.cvtColor(repaired_img, cv2.COLOR_BGR2RGB))
        plt.title("修复后图像"), plt.axis('off')
        
        # 显示修复前后对比
        plt.subplot(247), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("修复前（局部）"), plt.axis('off')
        if width > 400:  # 显示局部对比
            plt.xlim(width//2-100, width//2+100)
            plt.ylim(height//2-100, height//2+100)
        
        plt.subplot(248), plt.imshow(cv2.cvtColor(repaired_img, cv2.COLOR_BGR2RGB))
        plt.title("修复后（局部）"), plt.axis('off')
        if width > 400:
            plt.xlim(width//2-100, width//2+100)
            plt.ylim(height//2-100, height//2+100)
        
        plt.tight_layout()
        plt.show()
        
        # 保存结果
        cv2.imwrite("enhanced_detected.png", detected)
        cv2.imwrite("enhanced_repair_mask.png", repair_mask)
        cv2.imwrite("enhanced_repaired_image.png", repaired_img)
        
        print("修复完成！结果已保存")
        
    except Exception as e:
        print(f"错误：{e}")

# 如果您想要更激进的处理，可以使用这个版本
def ultra_enhanced_inpaint(original, repair_mask):
    """
    超强修复：结合多种技术
    """
    # 方法1：OpenCV修复
    repaired = cv2.inpaint(original, repair_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    # 方法2：高斯模糊混合（平滑过渡）
    blurred = cv2.GaussianBlur(repaired, (5, 5), 0)
    
    # 创建混合掩膜
    mask_expanded = cv2.dilate(repair_mask, np.ones((5, 5), np.uint8), iterations=2)
    mask_expanded = mask_expanded.astype(float) / 255.0
    
    # 混合原图和修复结果
    result = np.zeros_like(original, dtype=float)
    for c in range(3):  # 对每个颜色通道
        result[:, :, c] = (1 - mask_expanded) * original[:, :, c].astype(float) + \
                         mask_expanded * blurred[:, :, c].astype(float)
    
    return result.astype(np.uint8)

if __name__ == "__main__":
    main_enhanced()