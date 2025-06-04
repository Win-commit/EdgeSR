from sr_fusion import EdgeSRFusion
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import numpy as np
from important_patches import get_Important_concepts, get_patches, Incorporate

class CenterRegionAblationFusion(EdgeSRFusion):
    def detect_important_regions(self, img_path, topk=3, iou_threshold=0.75, containment_threshold=0.7):
        """
        消融实验：不做语义检测，直接返回图片中心1/4区域
        """
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        # 计算中心1/4区域
        x1 = w // 4
        y1 = h // 4
        x2 = w // 2
        y2 = h
        left_top_box = [x1, y1, x2, y2]
        return [left_top_box], {'left_top': [left_top_box]}

class HardFusionAblation(EdgeSRFusion):
    def process_image(self, img_path, save_path=None, visualize=False):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 1. 区域检测
        boxes, patches_dict = self.detect_important_regions(img_path)
        if not boxes:
            start_time = time.time()
            upsampled = cv2.resize(img, (img.shape[1]*self.scale, img.shape[0]*self.scale), interpolation=cv2.INTER_CUBIC)
            if save_path:
                cv2.imwrite(save_path, upsampled)
            return upsampled, None, time.time() - start_time

        # 2. 普通上采样整图
        start_time = time.time()
        h, w = img.shape[:2]
        upsampled_img = cv2.resize(img, (w*self.scale, h*self.scale), interpolation=cv2.INTER_CUBIC)
        sr_result = np.copy(upsampled_img)

        # 3. 对每个区域做超分，并直接替换到结果图像
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            region = img[y1:y2, x1:x2]
            if region.size == 0:
                continue
            temp_input = f"temp_region_input_{idx}.png"
            cv2.imwrite(temp_input, region)
            sr_region = self.sr_model.super_resolve(temp_input)
            os.remove(temp_input)
            sr_h, sr_w = sr_region.shape[:2]
            sr_result[y1*self.scale:y1*self.scale+sr_h, x1*self.scale:x1*self.scale+sr_w] = sr_region
        edge_sr_time = time.time() - start_time
        if save_path:
            cv2.imwrite(save_path, sr_result)
        return sr_result, None, edge_sr_time


class NoMergeRegionFusion(EdgeSRFusion):
    def detect_important_regions(self, img_path, topk=3, iou_threshold=0, containment_threshold=0):
        """
        消融实验：不合并区域，直接返回所有检测到的patch
        """
        # 获取重要概念
        concepts = get_Important_concepts(img_path, self.splicemodel, self.preprocess, self.vocabulary, topk)
        concepts_ls = list(concepts.keys())
        patches = get_patches(
            concepts_ls,
            img_path,
            self.cropper,
            IoU=iou_threshold,
            containment=containment_threshold,
            BOX_TRESHOLD=0.3,
            TEXT_TRESHOLD=0.25
        )
        all_boxes = []
        for concept, boxes in patches.items():
            all_boxes.extend(boxes)
        return all_boxes, patches

def find_hr_image(lr_path, hr_dir="/root/autodl-tmp/Datasets/DIV2K_valid_HR"):
        """根据低分辨率图像路径找到对应的高分辨率图像"""
        # 提取文件名，不含扩展名
        base_name = os.path.splitext(os.path.basename(lr_path))[0]
        # DIV2K数据集中LR图像名称通常以"x4"结尾，需要去除
        if base_name.endswith('x4'):
            base_name = base_name[:-2]
        
        # 构建HR图像路径
        # 尝试多种可能的扩展名
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
            hr_path = os.path.join(hr_dir, f"{base_name}{ext}")
            if os.path.exists(hr_path):
                return hr_path
        
        # 如果找不到对应的HR图像，返回None
        print(f"警告: 无法找到对应的HR图像: {base_name}")
        return None


def calculate_metrics(img1, img2):
        """计算两个图像之间的PSNR和SSIM指标"""
        # 确保图像尺寸相同
        if img1.shape != img2.shape:
            # 如果尺寸不同，将第一个图像调整为第二个图像的尺寸
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_CUBIC)
            
        # 计算PSNR
        psnr_value = psnr(img1, img2, data_range=255)
        
        # 计算SSIM
        # 转换为灰度图像用于SSIM计算
        if img1.ndim == 3:  # 彩色图像
            ssim_value = ssim(img1, img2, multichannel=True, data_range=255, channel_axis=2)
        else:  # 灰度图像
            ssim_value = ssim(img1, img2, data_range=255)
            
        return psnr_value, ssim_value

if __name__ == '__main__':
    LR_folder="/root/autodl-tmp/Datasets/Urban100/image_SRF_4/LR"
    HR_folder="/root/autodl-tmp/Datasets/Urban100/image_SRF_4/HR"
    # ablation = CenterRegionAblationFusion(sr_model_path="/root/EdgeSR/basemodels/basicSR/hat/HAT_SRx4_ImageNet-pretrain.pth", 
    #                                       model_type='HAT', 
    #                                       scale=4, 
    #                                       device='cuda')
    # ablation = HardFusionAblation(sr_model_path="/root/EdgeSR/basemodels/basicSR/hat/HAT_SRx4_ImageNet-pretrain.pth", 
    #                                       model_type='HAT', 
    #                                       scale=4, 
    #                                       device='cuda')
    ablation = NoMergeRegionFusion(sr_model_path="/root/EdgeSR/basemodels/basicSR/hat/HAT_SRx4_ImageNet-pretrain.pth", 
                                          model_type='HAT', 
                                          scale=4, 
                                          device='cuda')
    avg_time=0
    avg_psnr=0
    avg_ssim=0
    for lr_path in os.listdir(LR_folder):
        hr_path = find_hr_image(os.path.join(LR_folder, lr_path),hr_dir = HR_folder )
        if hr_path:
            fusion_result, _, edge_sr_time= ablation.process_image(img_path=os.path.join(LR_folder, lr_path),  visualize=False)
            hr_img = cv2.imread(hr_path)
            psnr_value, ssim_value = calculate_metrics(fusion_result, hr_img)
            print(f"Time: {edge_sr_time}, PSNR: {psnr_value}, SSIM: {ssim_value}")
            avg_time+=edge_sr_time
            avg_psnr+=psnr_value
            avg_ssim+=ssim_value
    print(f"Average Time: {avg_time/len(os.listdir(LR_folder))}, Average PSNR: {avg_psnr/len(os.listdir(LR_folder))}, Average SSIM: {avg_ssim/len(os.listdir(LR_folder))}")

