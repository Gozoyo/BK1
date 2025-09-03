import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_scratches_folds(image_path, sensitivity=50):
    """
    调整敏感度的划痕/折痕检测函数
    sensitivity: 敏感度（0-100），值越高检测越灵敏（可能增加误检）
    返回：原图，边缘图，过滤后边缘图，检测结果图，掩膜图，检测数量
    """
    # 读取图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("无法读取图像，请检查路径")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 1. 轻度去噪（避免过度去噪丢失细节）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 敏感度高时降低去噪强度
    h_value = 10 if sensitivity > 50 else 15
    denoised = cv2.fastNlMeansDenoising(blurred, h=h_value)
    
    # 2. 边缘检测（降低阈值，保留更多潜在边缘）
    threshold1 = max(10, 50 - sensitivity//2)  # 敏感度越高，阈值越低
    threshold2 = max(50, 100 - sensitivity)
    edges = cv2.Canny(denoised, threshold1, threshold2)
    
    # 3. 轻度形态学操作（只做一次膨胀，连接断续边缘）
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 4. 简化的方向过滤（只过滤方向高度一致的区域）
    filtered_edges = np.zeros_like(edges)
    # 计算梯度方向
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    edge_direction = np.arctan2(sobel_y, sobel_x)
    
    # 降低方向过滤的严格度
    dir_var_threshold = 0.05 + (sensitivity / 100) * 0.15  # 敏感度越高，阈值越低
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if edges[y, x] == 255:
                # 取5x5窗口（更大窗口提高稳定性）
                win_dir = edge_direction[y-2:y+3, x-2:x+3].flatten()
                win_dir = win_dir[np.abs(win_dir) > 0.1]  # 过滤接近0的方向
                if len(win_dir) < 5:  # 减少对窗口内边缘数量的要求
                    filtered_edges[y, x] = 255
                    continue
                # 计算方向方差
                dir_var = np.var(win_dir)
                if dir_var > dir_var_threshold:
                    filtered_edges[y, x] = 255
    
    # 5. 霍夫直线检测（放宽条件）
    min_line_length = max(10, 30 - sensitivity//2)  # 敏感度越高，最小长度越短
    max_line_gap = min(20, 10 + sensitivity//5)      # 敏感度越高，允许间隙越大
    
    lines = cv2.HoughLinesP(
        filtered_edges,
        rho=1, theta=np.pi/180,
        threshold=20,  # 降低阈值
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    # 创建掩膜图像（白色代表需要修复的区域）
    mask = np.zeros_like(gray)
    
    # 标记结果
    detected = original.copy()
    count = 0
    if lines is not None:
        count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 在检测结果图上画红线（显示用，线宽2）
            cv2.line(detected, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 在掩膜图上画白线（修复用，线宽比显示的要粗以确保完全覆盖缺陷）
            cv2.line(mask, (x1, y1), (x2, y2), (255), 3)
    
    return original, edges, filtered_edges, detected, mask, count

def inpaint_image(original_img, mask_img, method=cv2.INPAINT_TELEA):
    """
    使用OpenCV的inpaint函数修复图像
    :param original_img: 原始图像
    :param mask_img: 掩膜图像（白色为缺陷区域）
    :param method: 修复算法，cv2.INPAINT_TELEA 或 cv2.INPAINT_NS
    :return: 修复后的图像
    """
    # 确保掩膜是二值的（非黑即白）
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    
    # 进行修复
    inpainted_img = cv2.inpaint(original_img, binary_mask, inpaintRadius=3, flags=method)
    
    return inpainted_img

def main():
    image_path = "image3.jpg"  # 请替换为您的图像路径
    
    # 调整敏感度（建议从70开始测试，根据效果调整）
    sensitivity = 70  # 0-100，值越高检测越灵敏
    
    try:
        # 1. 检测划痕/折痕
        print("正在检测划痕和折痕...")
        original, edges, filtered, detected, mask, count = detect_scratches_folds(
            image_path, 
            sensitivity=sensitivity
        )
        
        print(f"检测到 {count} 处可能的划痕/折痕（敏感度：{sensitivity}）")
        
        # 2. 使用OpenCV进行修复
        print("正在进行图像修复...")
        repaired_img = inpaint_image(original, mask, cv2.INPAINT_TELEA)
        
        # 3. 保存所有结果
        cv2.imwrite("original.png", original)
        cv2.imwrite("edges_detection.png", edges)
        cv2.imwrite("filtered_edges.png", filtered)
        cv2.imwrite("detected_scratches.png", detected)
        cv2.imwrite("repair_mask.png", mask)
        cv2.imwrite("repaired_image.png", repaired_img)
        
        print("修复完成！结果已保存：")
        print("- detected_scratches.png: 检测结果（红色标记）")
        print("- repair_mask.png: 修复掩膜（白色区域将被修复）")
        print("- repaired_image.png: 修复后的图像")
        
        # 4. 显示所有结果
        plt.figure(figsize=(18, 12))
        
        plt.subplot(231), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原图"), plt.axis('off')
        
        plt.subplot(232), plt.imshow(edges, cmap="gray")
        plt.title("边缘检测结果"), plt.axis('off')
        
        plt.subplot(233), plt.imshow(filtered, cmap="gray")
        plt.title("过滤后边缘"), plt.axis('off')
        
        plt.subplot(234), plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
        plt.title(f"检测结果（{count}处）"), plt.axis('off')
        
        plt.subplot(235), plt.imshow(mask, cmap="gray")
        plt.title("修复掩膜"), plt.axis('off')
        
        plt.subplot(236), plt.imshow(cv2.cvtColor(repaired_img, cv2.COLOR_BGR2RGB))
        plt.title("修复后图像"), plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    main()