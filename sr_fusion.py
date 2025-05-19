# EdgeSR/sr_fusion.py
import cv2
import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
# 导入自定义模块
from important_patches import get_Important_concepts, get_patches, Incorporate
from sr_models import SuperResolutionModel
import SpLiCE.splice as splice
from openVS import PatchImages


class EdgeSRFusion:
    """边缘设备上的区域感知超分辨率融合模型"""
    
    def __init__(self, sr_model_path, model_type='DRCT', scale=4, device=None):
        """
        初始化融合超分模型
        
        Args:
            sr_model_path (str): 超分模型权重路径
            model_type (str): 超分模型类型
            scale (int): 超分倍数
            device (str): 使用的设备，'cuda'或'cpu'
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化超分模型
        self.scale = scale
        if model_type == 'DRCT':
            self.sr_model = SuperResolutionModel(
                model_type=model_type,
                model_path=sr_model_path,
                device=self.device,
                scale=scale,
                tile_size=512, 
                tile_overlap=32
            )
        else:
            self.sr_model = SuperResolutionModel(
                model_type=model_type,
                model_path=sr_model_path,
                device=self.device,
                scale=scale
            )
        # 初始化重要区域检测相关组件
        self.splicemodel = splice.load(
            "open_clip:ViT-B-32", 
            vocabulary="laion", 
            vocabulary_size=10000, 
            l1_penalty=0.15, 
            return_weights=True, 
            device=self.device
        )
        self.preprocess = splice.get_preprocess("open_clip:ViT-B-32")
        self.vocabulary = splice.get_vocabulary("laion", 10000)
        self.cropper = PatchImages(concepts=[])
        
        self.blend_width = 10  # 融合边缘的平滑宽度
        
    def detect_important_regions(self, img_path, topk=3, iou_threshold=0.75, containment_threshold=0.7):
        """检测图像中的重要区域，返回边界框"""
        # 获取重要概念
        concepts = get_Important_concepts(img_path, self.splicemodel, self.preprocess, self.vocabulary, topk)
        concepts_ls = list(concepts.keys())
        
        # 根据概念获取对应区域
        patches = get_patches(
            concepts_ls, 
            img_path, 
            self.cropper,
            IoU=iou_threshold,
            containment=containment_threshold,
            BOX_TRESHOLD=0.3,
            TEXT_TRESHOLD=0.25
        )
        
        # 将所有边界框合并成一个列表
        all_boxes = []
        for concept, boxes in patches.items():
            all_boxes.extend(boxes)
            
        return all_boxes, patches
    
    def create_region_mask(self, img, boxes):
        """根据边界框创建区域掩码，区分重要区域和非重要区域"""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 在掩码上标记重要区域
        for box in boxes:
            x1, y1, x2, y2 = box
            # 将坐标限制在图像边界内
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            mask[y1:y2, x1:x2] = 1.0
            
        # 创建边缘平滑过渡区域
        if self.blend_width > 0:
            mask = cv2.GaussianBlur(mask, (self.blend_width*2+1, self.blend_width*2+1), 0)
            
        return mask
    
    def process_image(self, img_path, save_path=None, visualize=False):
        """处理单张图像的完整流程"""
        # 1. 读取原图
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # 2. 检测重要区域
        boxes, patches_dict = self.detect_important_regions(img_path)
        if not boxes:
            print(f"未检测到重要区域，将对整张图片进行普通上采样")
            # 如果没有检测到重要区域，直接进行上采样
            start_time = time.time()
            upsampled = cv2.resize(img, (img.shape[1]*self.scale, img.shape[0]*self.scale), 
                                 interpolation=cv2.INTER_CUBIC)
            if save_path:
                cv2.imwrite(save_path, upsampled)
            return upsampled, None , time.time() - start_time # 没有区域划分
        
        #边缘端超分，开始计时
        start_time = time.time()
        # 3. 创建区域掩码
        region_mask = self.create_region_mask(img, boxes)
        
        # 4. 对整张图进行普通上采样
        h, w = img.shape[:2]
        upsampled_img = cv2.resize(img, (w*self.scale, h*self.scale), interpolation=cv2.INTER_CUBIC)
        
        # 5. 创建SR结果图像画布
        sr_result = np.copy(upsampled_img)
        
        # 6. 对每个重要区域进行超分辨率处理
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 提取区域
            region = img[y1:y2, x1:x2]
            if region.size == 0:  # 跳过空区域
                continue
                
            # 保存临时区域
            temp_input = f"temp_region_input_{idx}.png"
            cv2.imwrite(temp_input, region)
            
            # 对区域进行超分辨率处理
            sr_region = self.sr_model.super_resolve(temp_input)
            
            # 删除临时文件
            os.remove(temp_input)
            
            # 将超分结果放入结果图像中的对应位置
            sr_h, sr_w = sr_region.shape[:2]
            sr_result[y1*self.scale:y1*self.scale+sr_h, x1*self.scale:x1*self.scale+sr_w] = sr_region
        
        # 7. 对掩码也进行上采样
        upsampled_mask = cv2.resize(region_mask, (w*self.scale, h*self.scale), interpolation=cv2.INTER_LINEAR)
        
        # 8. 根据掩码融合SR区域和普通上采样区域
        upsampled_mask = np.stack([upsampled_mask] * 3, axis=2)  # 扩展到3通道
        fusion_result = upsampled_mask * sr_result + (1 - upsampled_mask) * upsampled_img
        fusion_result = fusion_result.astype(np.uint8)
        
        #边缘端超分，结束计时
        edge_sr_time = time.time() - start_time
        
        # 9. 可视化结果（如果需要）
        whole_img_sr = None
        if visualize:
            # 保存临时整图
            temp_whole_img = "temp_whole_img.png"
            cv2.imwrite(temp_whole_img, img)
            
            # 对整图进行超分
            whole_img_sr = self.sr_model.super_resolve(temp_whole_img)
            
            # 删除临时文件
            os.remove(temp_whole_img)

    # 修改可视化部分为2x3布局
        if visualize:
            # 在原图上绘制边界框
            vis_img = np.copy(img)
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 创建可视化图像
            plt.figure(figsize=(18, 12))
            
            plt.subplot(2, 3, 1)
            plt.title("Original image with detection area")
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            
            plt.subplot(2, 3, 2)
            plt.title("Critical area mask")
            plt.imshow(region_mask, cmap='jet')
            
            plt.subplot(2, 3, 3)
            plt.title("Ordinary up-sampling results")
            plt.imshow(cv2.cvtColor(upsampled_img, cv2.COLOR_BGR2RGB))
            
            plt.subplot(2, 3, 4)
            plt.title("Full image SR results")
            plt.imshow(cv2.cvtColor(whole_img_sr, cv2.COLOR_BGR2RGB))
            
            plt.subplot(2, 3, 5)
            plt.title("Partial SR + fusion results")
            plt.imshow(cv2.cvtColor(fusion_result, cv2.COLOR_BGR2RGB))
            
            # 添加空白子图或可选的PSNR/SSIM对比
            plt.subplot(2, 3, 6)
            plt.title("Results comparison")
            plt.axis('off')
            # 可以在这里添加额外的比较信息，如处理时间对比等
            
            plt.tight_layout()
            vis_path = os.path.splitext(save_path)[0] + "_visualization.png" if save_path else "sr_fusion_visualization.png"
            plt.savefig(vis_path, dpi=300)
            plt.close()
            
        # 10. 保存结果
        if save_path:
            cv2.imwrite(save_path, fusion_result)
        
        return fusion_result, region_mask, edge_sr_time
    
    def batch_process(self, input_dir, output_dir, visualize=False):
        """批量处理文件夹中的图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历文件夹中的所有图像
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for idx, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_SR_{self.scale}x.png")
            
            print(f"处理图像 {idx+1}/{len(image_files)}: {filename}")
            self.process_image(input_path, output_path, visualize)


if __name__ == "__main__":
    # 示例用法
    sr_fusion = EdgeSRFusion(
        sr_model_path="/root/EdgeSR/basemodels/DRCT/.ckpts/DRCT-L_X4.pth",
        model_type="DRCT",
        scale=4
    )
    
    # 处理单张图像
    sr_fusion.process_image(
        img_path="/root/autodl-tmp/Datasets/DIV2K_valid_LR_bicubic/X4/0801x4.png",
        save_path="output/result.png",
        visualize=True
    )
    
    # 或者批量处理
    # sr_fusion.batch_process(
    #     input_dir="/path/to/input/folder",
    #     output_dir="/path/to/output/folder",
    #     visualize=True
    # )