import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_damage(image_path):
    """
    使用边缘检测法和滤波差异法检测图像中的损坏区域（改进+升级版）
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像文件 '{image_path}'")
        return None, None, None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 边缘检测 ---
    def edge_detection(gray_img, low_threshold=50, high_threshold=150):
        edges = cv2.Canny(gray_img, low_threshold, high_threshold)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=2)
        return dilated

    # --- 滤波差异 ---
    def filtering_difference(gray_img, kernel_size=7, threshold=30):
        blurred = cv2.medianBlur(gray_img, kernel_size)
        diff = cv2.absdiff(gray_img, blurred)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    edge_mask = edge_detection(gray_blur)
    filter_mask = filtering_difference(gray_blur)

    edge_mask = remove_small_areas(edge_mask, min_area=100)
    filter_mask = remove_small_areas(filter_mask, min_area=100)

    combined_mask = cv2.bitwise_or(edge_mask, filter_mask)
    final_mask = cv2.dilate(combined_mask, np.ones((5, 5), np.uint8), iterations=1)
    final_mask = remove_small_areas(final_mask, min_area=200)

    return gray, edge_mask, filter_mask, final_mask, image

def remove_small_areas(mask, min_area=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[labels == i] = 255
    return new_mask

def detect_and_label_defects(original, mask, output_path="highlighted_labeled.jpg"):
    """
    根据 final_mask 检测坏点，标注编号
    """
    # 转换为轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = original.copy()
    defect_count = 0

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        # --- 筛选坏点条件 ---
        if area > 200 and aspect_ratio > 2:  
            defect_count += 1
            # 绘制矩形框
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 编号标注
            cv2.putText(annotated, f"ID {defect_count}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, annotated)
    print(f"坏点检测完成！共检测到 {defect_count} 个坏点，结果已保存为 {output_path}")
    return annotated

def display_results(original, edge_mask, filter_mask, combined_mask):
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
    plt.savefig('damage_detection_results_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    image_path = "image2.png"
    gray, edge_mask, filter_mask, final_mask, original_color = detect_damage(image_path)
    
    if gray is not None:
        display_results(gray, edge_mask, filter_mask, final_mask)
        
        # 检测并标注坏点
        annotated = detect_and_label_defects(original_color, final_mask, "highlighted_damage_labeled.jpg")
        
        # 保存掩模
        cv2.imwrite("edge_mask_labeled.jpg", edge_mask)
        cv2.imwrite("filter_mask_labeled.jpg", filter_mask)
        cv2.imwrite("final_mask_labeled.jpg", final_mask)

        print("处理完成! 已生成以下文件:")
        print("- damage_detection_results_labeled.png: 改进后的对比图")
        print("- edge_mask_labeled.jpg: 边缘检测掩模")
        print("- filter_mask_labeled.jpg: 滤波差异掩模")
        print("- final_mask_labeled.jpg: 最终掩模")
        print("- highlighted_damage_labeled.jpg: 在原图上自动框出坏点并编号")
