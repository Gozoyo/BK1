import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_damage(image_path):
    """
    使用边缘检测法和滤波差异法检测图像中的损坏区域（改进版）
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像文件 '{image_path}'")
        return None, None, None, None
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 新增：高斯模糊，减少自然纹理干扰 ---
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 边缘检测法 (检测划痕、裂缝)
    def edge_detection(gray_img, low_threshold=50, high_threshold=150):
        edges = cv2.Canny(gray_img, low_threshold, high_threshold)
        
        # 形态学操作 - 闭操作，连接断开的边缘
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 膨胀操作，使坏点更粗
        dilated = cv2.dilate(closed, kernel, iterations=2)
        
        return dilated

    # 3. 滤波差异法 (检测灰尘、霉斑等)
    def filtering_difference(gray_img, kernel_size=7, threshold=30):
        blurred = cv2.medianBlur(gray_img, kernel_size)
        diff = cv2.absdiff(gray_img, blurred)
        
        # 提高阈值，避免误检背景
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 开运算去噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 闭运算填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    # 4. 应用两种方法
    edge_mask = edge_detection(gray_blur)
    filter_mask = filtering_difference(gray_blur)

    # --- 新增：分别去掉小区域 ---
    edge_mask = remove_small_areas(edge_mask, min_area=100)
    filter_mask = remove_small_areas(filter_mask, min_area=100)

    # 5. 合并两种方法结果
    combined_mask = cv2.bitwise_or(edge_mask, filter_mask)

    # 6. 后处理 - 膨胀 + 连通域去小块 + 填充
    final_mask = cv2.dilate(combined_mask, np.ones((5, 5), np.uint8), iterations=1)
    final_mask = remove_small_areas(final_mask, min_area=200)

    return gray, edge_mask, filter_mask, final_mask

def remove_small_areas(mask, min_area=50):
    """去除小面积区域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # 跳过背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[labels == i] = 255
    return new_mask

def display_results(original, edge_mask, filter_mask, combined_mask):
    """显示处理结果"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('原始灰度图像')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(edge_mask, cmap='gray')
    plt.title('边缘检测法 - 线性瑕疵')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(filter_mask, cmap='gray')
    plt.title('滤波差异法 - 颗粒状瑕疵')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(combined_mask, cmap='gray')
    plt.title('最终掩模')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('damage_detection_results_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

def apply_mask_to_image(original, mask, output_path):
    """将掩模应用到原始图像上，突出显示损坏区域"""
    color_mask = np.zeros_like(original)
    color_mask[mask == 255] = [0, 0, 255]  # 红色
    highlighted = cv2.addWeighted(original, 0.7, color_mask, 0.3, 0)
    cv2.imwrite(output_path, highlighted)
    print(f"高亮显示损坏区域的结果已保存为: {output_path}")
    return highlighted

# 主程序
if __name__ == "__main__":
    image_path = "image1.jpg"
    gray, edge_mask, filter_mask, final_mask = detect_damage(image_path)
    
    if gray is not None:
        display_results(gray, edge_mask, filter_mask, final_mask)
        original_color = cv2.imread(image_path)
        highlighted_image = apply_mask_to_image(original_color, final_mask, "highlighted_damage_improved.jpg")
        
        # 保存各个掩模
        cv2.imwrite("edge_mask_improved.jpg", edge_mask)
        cv2.imwrite("filter_mask_improved.jpg", filter_mask)
        cv2.imwrite("final_mask_improved.jpg", final_mask)
        
        print("处理完成! 已生成以下文件:")
        print("- damage_detection_results_improved.png: 改进后的对比图")
        print("- edge_mask_improved.jpg: 改进的边缘检测掩模")
        print("- filter_mask_improved.jpg: 改进的滤波差异掩模")
        print("- final_mask_improved.jpg: 改进的最终掩模")
        print("- highlighted_damage_improved.jpg: 改进的高亮结果")
