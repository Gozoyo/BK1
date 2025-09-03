import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_damage(image_path):
    """
    使用边缘检测法和滤波差异法检测图像中的损坏区域
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像文件 '{image_path}'")
        return None, None, None, None
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 边缘检测法 (用于检测划痕、裂缝等线性瑕疵)
    def edge_detection(gray_img, low_threshold=50, high_threshold=150):
        # 使用Canny边缘检测
        edges = cv2.Canny(gray_img, low_threshold, high_threshold)
        
        # 形态学操作 - 闭操作，连接断开的边缘
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 膨胀操作，使边缘更加明显
        dilated_edges = cv2.dilate(closed_edges, kernel, iterations=1)
        
        return dilated_edges
    
    # 3. 滤波差异法 (用于检测灰尘、霉斑等颗粒状瑕疵)
    def filtering_difference(gray_img, kernel_size=5, threshold=20):
        # 应用中值滤波，去除小噪点
        blurred = cv2.medianBlur(gray_img, kernel_size)
        
        # 计算原图与模糊图的绝对差异
        diff = cv2.absdiff(gray_img, blurred)
        
        # 将差异图二值化
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作 - 开操作，去除小噪点
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return cleaned_mask
    
    # 4. 应用两种方法
    edge_mask = edge_detection(gray)
    filter_mask = filtering_difference(gray)
    
    # 5. 合并两种方法的掩模
    combined_mask = cv2.bitwise_or(edge_mask, filter_mask)
    
    # 6. 后处理 - 去除小面积区域
    def remove_small_areas(mask, min_area=50):
        # 寻找连通区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 创建一个新掩模，只保留大面积区域
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # 跳过背景(0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                new_mask[labels == i] = 255
                
        return new_mask
    
    final_mask = remove_small_areas(combined_mask, min_area=50)
    
    return gray, edge_mask, filter_mask, final_mask

def display_results(original, edge_mask, filter_mask, combined_mask):
    """
    显示处理结果
    """
    # 使用matplotlib显示结果
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('原始灰度图像')
    plt.axis('off')
    
    # 边缘检测结果
    plt.subplot(2, 2, 2)
    plt.imshow(edge_mask, cmap='gray')
    plt.title('边缘检测法 - 线性瑕疵')
    plt.axis('off')
    
    # 滤波差异结果
    plt.subplot(2, 2, 3)
    plt.imshow(filter_mask, cmap='gray')
    plt.title('滤波差异法 - 颗粒状瑕疵')
    plt.axis('off')
    
    # 合并结果
    plt.subplot(2, 2, 4)
    plt.imshow(combined_mask, cmap='gray')
    plt.title('合并后的最终掩模')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('damage_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def apply_mask_to_image(original, mask, output_path):
    """
    将掩模应用到原始图像上，突出显示损坏区域
    """
    # 创建彩色掩模 (红色)
    color_mask = np.zeros_like(original)
    color_mask[mask == 255] = [0, 0, 255]  # 红色
    
    # 将彩色掩模与原始图像叠加
    highlighted = cv2.addWeighted(original, 0.7, color_mask, 0.3, 0)
    
    # 保存结果
    cv2.imwrite(output_path, highlighted)
    print(f"高亮显示损坏区域的结果已保存为: {output_path}")
    
    return highlighted

# 主程序
if __name__ == "__main__":
    # 修改图像路径为 image1.jpg
    image_path = "image1.jpg"
    
    # 检测损坏区域
    gray, edge_mask, filter_mask, final_mask = detect_damage(image_path)
    
    if gray is not None:
        # 显示结果
        display_results(gray, edge_mask, filter_mask, final_mask)
        
        # 将彩色掩模应用到原始图像
        original_color = cv2.imread(image_path)
        highlighted_image = apply_mask_to_image(original_color, final_mask, "highlighted_damage.jpg")
        
        # 保存各个掩模
        cv2.imwrite("edge_mask.jpg", edge_mask)
        cv2.imwrite("filter_mask.jpg", filter_mask)
        cv2.imwrite("final_mask.jpg", final_mask)
        
        print("处理完成! 已生成以下文件:")
        print("- damage_detection_results.png: 所有结果的对比图")
        print("- edge_mask.jpg: 边缘检测法生成的掩模")
        print("- filter_mask.jpg: 滤波差异法生成的掩模")
        print("- final_mask.jpg: 合并后的最终掩模")
        print("- highlighted_damage.jpg: 在原图上高亮显示损坏区域")