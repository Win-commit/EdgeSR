# EdgeSR/sr_models.py
import argparse
import cv2
import glob
import numpy as np
import os
import torch
import sys
sys.path.append('basemodels/DRCT')
from drct.archs.DRCT_arch import *

class SuperResolutionModel:
    """通用的超分辨率模型接口，用于统一不同底层模型的API"""
    
    def __init__(self, model_type='DRCT', model_path=None, device=None, scale=4, tile_size=None, tile_overlap=32, window_size=16):
        """
        初始化超分模型
        
        Args:
            model_type (str): 模型类型，如'DRCT'
            model_path (str): 模型权重路径
            device (str): 使用的设备，如'cuda'或'cpu'
            scale (int): 超分倍数
            tile_size (int): 分块处理的瓦片大小，None表示整图处理
            tile_overlap (int): 瓦片重叠区域大小
            window_size (int): 窗口大小（某些模型需要）
        """
        self.model_type = model_type
        self.model_path = model_path
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.window_size = window_size
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self):
        """加载指定类型的模型"""
        if self.model_type == 'DRCT':
            # 设置DRCT模型（可以根据需要调整参数）
            model = DRCT(upscale=self.scale, in_chans=3, img_size=64, window_size=self.window_size, 
                        compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, 
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 
                        gc=32, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            
            # 加载预训练权重
            if self.model_path is not None:
                model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
            
            model.eval()
            return model.to(self.device)
        else:
            raise NotImplementedError(f"模型类型 {self.model_type} 尚未实现")
    
    def preprocess(self, img_path):
        """预处理输入图像"""
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # 根据窗口大小进行填充（仅对特定模型如DRCT）
        if self.model_type == 'DRCT':
            _, _, h_old, w_old = img.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
            
            # 记录原始尺寸，用于后处理
            self.original_size = (h_old, w_old)
            
        return img
    
    def inference(self, img):
        """执行超分推理"""
        if self.tile_size is None:
            # 整图处理
            output = self.model(img)
        else:
            # 分块处理
            b, c, h, w = img.size()
            tile = min(self.tile_size, h, w)
            
            # 确保tile大小是window_size的倍数
            if self.model_type == 'DRCT':
                assert tile % self.window_size == 0, "tile size should be a multiple of window_size"
                
            stride = tile - self.tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*self.scale, w*self.scale).type_as(img)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = self.model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*self.scale:(h_idx+tile)*self.scale, 
                       w_idx*self.scale:(w_idx+tile)*self.scale].add_(out_patch)
                    W[..., h_idx*self.scale:(h_idx+tile)*self.scale, 
                       w_idx*self.scale:(w_idx+tile)*self.scale].add_(out_patch_mask)
            
            output = E.div_(W)
        
        return output
    
    def postprocess(self, output):
        """后处理输出图像"""
        # 裁剪到原始尺寸（仅对特定模型如DRCT）
        if self.model_type == 'DRCT' and hasattr(self, 'original_size'):
            h_old, w_old = self.original_size
            output = output[..., :h_old * self.scale, :w_old * self.scale]
            
        # 转换为numpy数组并保存
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        return output
    
    def super_resolve(self, img_path, save_path=None):
        """执行完整的超分辨率过程"""
        try:
            # 预处理
            img = self.preprocess(img_path)
            
            # 推理
            with torch.no_grad():
                output = self.inference(img)
            
            # 后处理
            result = self.postprocess(output)
            
            # 保存结果
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result)
            
            return result
        
        except Exception as e:
            print(f"超分辨率过程出错: {str(e)}")
            return None
    
    def batch_process(self, input_dir, output_dir, name_suffix='_SR'):
        """批量处理文件夹中的图片"""
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, path in enumerate(sorted(glob.glob(os.path.join(input_dir, '*')))):
            # 跳过非图像文件
            if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
                
            imgname = os.path.splitext(os.path.basename(path))[0]
            print(f'处理第 {idx} 张图片: {imgname}')
            
            # 设置输出路径
            output_path = os.path.join(output_dir, f'{imgname}{name_suffix}_X{self.scale}.png')
            
            # 执行超分
            self.super_resolve(path, output_path)