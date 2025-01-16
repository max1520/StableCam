import os
import cv2
import torch
import lpips
import numpy as np

def calculate_lpips_folder(image_dir1, image_dir2):
    # List all images in the directories
    images1 = sorted(os.listdir(image_dir1))
    images2 = sorted(os.listdir(image_dir2))

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='vgg',spatial=False)  # You can also use 'alex' or 'squeeze' for different networks
    lpips_model.eval()  # Set model to evaluation mode

    total_lpips = 0
    num_images = len(images1)

    for img1_name, img2_name in zip(images1, images2):
        img1_path = os.path.join(image_dir1, img1_name)
        img2_path = os.path.join(image_dir2, img2_name)

        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

        # Convert to RGB and normalize to [0, 1]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.0

        # Ensure images are in the correct shape (C, H, W)
        img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float()
        img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float()

        # Compute LPIPS
        lpips_value = lpips_model(img1, img2)  # Returns a tensor with the LPIPS score
        lpips_value = lpips_value.item()  # Convert tensor to scalar

        total_lpips += lpips_value
        print(f"{img1_name} - LPIPS: {lpips_value}")

    # Calculate average LPIPS
    avg_lpips = total_lpips / num_images
    print(f"\nAverage LPIPS: {avg_lpips}")


if __name__ == '__main__':
    choice = 'stableSR-calibration_SFT'
    if choice == 'stableSR-calibration_SFT':
        image_dir1 = r"D:\cqy\stableSR\CFW_data\eval_data\amplitude_512_calibration\samples"
        image_dir2 = r"D:\cqy\stableSR\CFW_data\eval_data\amplitude_512_calibration\gts"


    calculate_lpips_folder(image_dir1, image_dir2)