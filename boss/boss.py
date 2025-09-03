import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt

def detect_scratches_deep_learning(image_path):
    # 1. 加载图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. 加载预训练的语义分割模型（这里用DeepLabv3，可替换为其他分割模型）
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()  # 设为评估模式
    
    # 3. 图像预处理（匹配模型输入要求）
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img_rgb).unsqueeze(0)  # 增加batch维度
    
    # 4. 模型推理
    with torch.no_grad():
        output = model(input_tensor)['out']  # 模型输出的分割结果
    
    # 5. 后处理：获取“可能的损伤”区域（这里简化，实际需针对老照片微调类别）
    # DeepLabv3预训练在COCO数据集，没有“划痕”类别，所以这里演示“边缘类”区域
    # 若有自己的训练数据，需替换为自定义模型
    output_predictions = output.argmax(1).squeeze(0).numpy()
    # 假设“边缘/线条类”属于类别23（需根据实际预训练数据集调整）
    scratch_mask = (output_predictions == 23).astype(np.uint8) * 255
    
    # 6. 形态学操作：优化mask
    kernel = np.ones((3, 3), np.uint8)
    scratch_mask = cv2.dilate(scratch_mask, kernel, iterations=1)
    scratch_mask = cv2.erode(scratch_mask, kernel, iterations=1)
    
    # 7. 标记原图
    marked_img = img.copy()
    marked_img[scratch_mask > 0] = [0, 0, 255]  # 红色标记
    
    return img_rgb, scratch_mask, marked_img

def main():
    image_path = "image2.png"  # 替换为你的图像路径
    try:
        original, mask, marked = detect_scratches_deep_learning(image_path)
        
        # 保存结果
        cv2.imwrite("deep_learning_mask.png", mask)
        cv2.imwrite("deep_learning_marked.png", cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
        
        # 显示结果
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(original), plt.title("原图")
        plt.subplot(132), plt.imshow(mask, cmap="gray"), plt.title("损伤掩码")
        plt.subplot(133), plt.imshow(marked), plt.title("标记结果")
        plt.tight_layout(), plt.show()
    except Exception as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    main()