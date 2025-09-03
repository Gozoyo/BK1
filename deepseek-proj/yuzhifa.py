import cv2
import numpy as np

# 1. 读取图像文件
# 替换 'image1.jpg' 为您的实际文件路径和名称
image_path = 'image1.jpg'
image = cv2.imread(image_path)

# 检查图像是否成功加载
if image is None:
    print(f"错误: 无法读取图像文件 '{image_path}'")
    print("请检查文件路径是否正确，以及文件是否存在")
    exit()

# 2. 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 应用阈值函数
def apply_threshold(gray_img, thresh_value=200):
    # 用 THRESH_BINARY_INV：将暗于阈值的像素（划痕）变为白色(255)，亮于阈值的变为黑色(0)
    _, mask_binary = cv2.threshold(gray_img, thresh_value, 255, cv2.THRESH_BINARY_INV)
    return mask_binary

# 4. 尝试不同的阈值并显示结果
# 创建一个窗口用于显示图像
cv2.namedWindow('Results', cv2.WINDOW_NORMAL)

# 尝试不同的阈值
for thresh in [150, 175, 200, 225]:
    mask = apply_threshold(gray_image, thresh)
    
    # 将原图和掩模并排显示
    result = np.hstack((gray_image, mask))
    
    # 显示图像
    cv2.imshow('Results', result)
    cv2.resizeWindow('Results', 1000, 500)  # 调整窗口大小
    
    print(f"按任意键查看下一个阈值 (当前阈值: {thresh})")
    cv2.waitKey(0)  # 等待按键

# 5. 保存最终选择的掩模
# 选择效果最好的阈值，这里以200为例
final_threshold = 200
final_mask = apply_threshold(gray_image, final_threshold)

# 保存掩模图像
cv2.imwrite('damage_mask.png', final_mask)
print(f"掩模已保存为 'damage_mask.png'")

# 6. 关闭所有窗口
cv2.destroyAllWindows()
