import os
import cv2
import numpy as np

# 读取文件夹内图片的函数
def get_image_files(folder):
    image_files = []
    for file_name in os.listdir(folder):
        if file_name.lower().endswith('.png'):  # 假设图像格式为 .png
            image_files.append(os.path.join(folder, file_name))

    # 按照文件名中的数字进行排序
    image_files.sort(key=lambda x: int(x.split('(')[-1].split(')')[0]))
    return image_files

# 计算两个向量的余弦相似度
def cosine_similarity(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # 确保不出现零向量
    if np.linalg.norm(img1_flat) == 0 or np.linalg.norm(img2_flat) == 0:
        return 0.0

    cos_sim = np.dot(img1_flat, img2_flat) / (np.linalg.norm(img1_flat) * np.linalg.norm(img2_flat))
    return cos_sim

# 计算两个文件夹中所有对应图像的余弦距离
def calculate_cosine_distances(folder_1, folder_2):
    image_files_1 = get_image_files(folder_1)
    image_files_2 = get_image_files(folder_2)

    assert len(image_files_1) == len(image_files_2), "两个文件夹中的图像数量不匹配"

    cos_similarities = []

    # 遍历每对图像，计算余弦相似度
    for img1_path, img2_path in zip(image_files_1, image_files_2):
        # 读取图像，保持彩色图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # 检查图像是否读取成功
        if img1 is None or img2 is None:
            print(f"Error loading image: {img1_path} or {img2_path}")
            continue

        # 将图像缩放到相同尺寸
        img1_resized = cv2.resize(img1, (256, 256))  # 你可以根据需求调整大小
        img2_resized = cv2.resize(img2, (256, 256))

        # 转换为浮动类型，并将像素值归一化到 [0, 1]
        img1_resized = img1_resized.astype(np.float32) / 255.0
        img2_resized = img2_resized.astype(np.float32) / 255.0

        # 计算余弦相似度
        cos_sim = cosine_similarity(img1_resized, img2_resized)
        cos_similarities.append(cos_sim)

        # print(f"Cosine similarity between {os.path.basename(img1_path)} and {os.path.basename(img2_path)}: {cos_sim:.4f}")
        print(f"{cos_sim:.2f}")

    # 输出平均余弦相似度
    avg_cos_sim = np.mean(cos_similarities)
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")

# 设置文件夹路径
# folder_1 = r"D:\cqy\stableSR\ablation_study\intermediates_spade_results\intermediates"
# folder_2 = r"D:\cqy\stableSR\ablation_study\intermediates_spadezero_results\intermediates"
folder_1 = r"D:\cqy\stableSR\ablation_study\intermediates_noT_spade_results\intermediates"
folder_2 = r"D:\cqy\stableSR\ablation_study\intermediates_noT_spadezero_results\intermediates"

# 调用计算函数
calculate_cosine_distances(folder_1, folder_2)
