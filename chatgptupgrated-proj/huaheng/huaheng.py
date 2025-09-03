import cv2
import numpy as np

# 读取图像
img = cv2.imread("image1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 阈值分割（提取非常暗的区域 = 黑色划痕）
_, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

# 形态学操作：去噪点，连接裂纹
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上画出坏点（黑色划痕）
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:  # 过滤小噪点，只保留较大划痕
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# 保存结果
cv2.imwrite("scratches_detected.png", img)
print("检测完成！坏点已标红，结果保存在 scratches_detected.png")
