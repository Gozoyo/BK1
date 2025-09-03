import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_scratches_folds(image_path, sensitivity=50):
    # 读取图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("无法读取图像，请检查路径")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 1. 轻度去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    h_value = 10 if sensitivity > 50 else 15
    denoised = cv2.fastNlMeansDenoising(blurred, h=h_value)

    # 2. 边缘检测（降低阈值，保留更多潜在边缘）
    threshold1 = max(10, 50 - sensitivity // 2)
    threshold2 = max(50, 100 - sensitivity)
    edges = cv2.Canny(denoised, threshold1, threshold2)

    # 3. 轻度形态学操作（连接断续边缘）
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 4. 方向与长度过滤（重点：区分线性损伤和面部细节）
    filtered_edges = np.zeros_like(edges)
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    edge_direction = np.arctan2(sobel_y, sobel_x)
    dir_var_threshold = 0.05 + (sensitivity / 100) * 0.15

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if edges[y, x] == 255:
                win_dir = edge_direction[y-2:y+3, x-2:x+3].flatten()
                win_dir = win_dir[np.abs(win_dir) > 0.1]
                if len(win_dir) < 5:
                    filtered_edges[y, x] = 255
                    continue
                dir_var = np.var(win_dir)
                # 新增：长度过滤（面部细节更短，划痕更长）
                # 计算3x3窗口内边缘像素数量，间接反映长度
                edge_count = np.sum(edges[y-1:y+2, x-1:x+2] == 255)
                if dir_var > dir_var_threshold and edge_count > 3:  # 边缘数量>3才保留
                    filtered_edges[y, x] = 255

    # 5. 霍夫直线检测（更严格，过滤短线条）
    min_line_length = max(20, 30 - sensitivity // 2)  # 最小长度从20开始，减少短线条误检
    max_line_gap = min(15, 10 + sensitivity // 5)
    lines = cv2.HoughLinesP(
        filtered_edges,
        rho=1, theta=np.pi/180,
        threshold=30,  # 提高阈值，减少候选直线
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # 6. 面部区域过滤（可选：用人脸检测排除面部误检）
    # 初始化人脸检测器（需确保OpenCV版本支持，若报错可注释此部分）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_mask = np.ones_like(gray) * 255  # 人脸区域设为255（保留），其他为0（过滤）
    for (x, y, w, h) in faces:
        cv2.rectangle(face_mask, (x, y), (x + w, y + h), 0, -1)  # 人脸区域画0，排除

    # 标记结果
    detected = original.copy()
    count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 检查线段是否在人脸区域外（或注释此if，仅靠长度过滤）
            if face_mask[y1, x1] == 255 and face_mask[y2, x2] == 255:
                cv2.line(detected, (x1, y1), (x2, y2), (0, 0, 255), 2)
                count += 1

    return original, edges, filtered_edges, detected, count

def main():
    image_path = "image3.jpg"
    sensitivity = 60  # 降低敏感度，减少误检

    try:
        original, edges, filtered, detected, count = detect_scratches_folds(
            image_path, 
            sensitivity=sensitivity
        )

        print(f"检测到 {count} 处可能的划痕/折痕（敏感度：{sensitivity}）")
        cv2.imwrite("adjusted_detected.png", detected)

        plt.figure(figsize=(16, 4))
        plt.subplot(141), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), plt.title("原图")
        plt.subplot(142), plt.imshow(edges, cmap="gray"), plt.title("边缘检测结果")
        plt.subplot(143), plt.imshow(filtered, cmap="gray"), plt.title("过滤后边缘")
        plt.subplot(144), plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)), plt.title(f"检测结果（{count}处）")
        plt.tight_layout(), plt.show()

    except Exception as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    main()