# EdgeSR/experiments.py
import os
import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 导入自定义模块
from sr_fusion import EdgeSRFusion

class SRExperiment:
    """超分辨率方法实验评估类"""
    
    def __init__(self, sr_model_path, model_type='DRCT', scale=4, device=None):
        """
        初始化实验环境
        
        Args:
            sr_model_path (str): 超分模型权重路径
            model_type (str): 超分模型类型
            scale (int): 超分倍数
            device (str): 使用的设备，'cuda'或'cpu'
        """
        # 创建区域感知超分融合模型
        self.sr_fusion = EdgeSRFusion(
            sr_model_path=sr_model_path,
            model_type=model_type,
            scale=scale,
            device=device
        )
        
        # 直接创建超分模型（用于整图超分比较）
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model_type = model_type
        self.scale = scale
        self.sr_model_path = sr_model_path
    
    def calculate_metrics(self, img1, img2):
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
    
    def find_hr_image(self, lr_path, hr_dir="/root/autodl-tmp/Datasets/Urban100/image_SRF_4/HR"):
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
    
    def run_single_image_experiment(self, lr_path, output_dir, visualize=True):
        """对单张图像进行实验对比"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件名处理
        filename = os.path.basename(lr_path)
        base_filename = os.path.splitext(filename)[0]
        
        # 计时开始
        start_time_total = time.time()
        
        # 1. 读取LR图像
        lr_img = cv2.imread(lr_path)
        if lr_img is None:
            raise ValueError(f"无法读取图像: {lr_path}")
        h, w = lr_img.shape[:2]
        
        # 2. 找到对应的HR图像（真值）
        hr_path = self.find_hr_image(lr_path)
        hr_img = None
        if hr_path:
            hr_img = cv2.imread(hr_path)
            # 裁剪HR图像，使其尺寸与放大后的LR图像匹配
            hr_img = cv2.resize(hr_img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_AREA)
        
        # 3. 双三次上采样（基准方法）
        start_time_bicubic = time.time()
        bicubic_img = cv2.resize(lr_img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
        time_bicubic = time.time() - start_time_bicubic
        
        # 4. 整图超分处理
        start_time_full = time.time()
        temp_lr = os.path.join(output_dir, f"temp_{base_filename}.png")
        cv2.imwrite(temp_lr, lr_img)
        full_sr_img = self.sr_fusion.sr_model.super_resolve(temp_lr)
        os.remove(temp_lr)
        time_full_sr = time.time() - start_time_full
        
        # 5. 区域感知超分 + 融合 (使用EdgeSRFusion的process_image方法)
        # start_time_partial = time.time()
        partial_sr_output_path = os.path.join(output_dir, f"{base_filename}_partial_sr.png")
        partial_sr_img, region_mask, time_partial_sr = self.sr_fusion.process_image(
            lr_path, 
            save_path=partial_sr_output_path, 
            visualize=False  
        )
        # time_partial_sr = time.time() - start_time_partial
        
        # 6. 保存全图超分和双三次上采样结果
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_bicubic.png"), bicubic_img)
        cv2.imwrite(os.path.join(output_dir, f"{base_filename}_full_sr.png"), full_sr_img)
        
        # 7. 计算评价指标
        metrics = {}
        if hr_img is not None:
            # 计算PSNR和SSIM
            metrics['bicubic_psnr'], metrics['bicubic_ssim'] = self.calculate_metrics(bicubic_img, hr_img)
            metrics['full_sr_psnr'], metrics['full_sr_ssim'] = self.calculate_metrics(full_sr_img, hr_img)
            metrics['partial_sr_psnr'], metrics['partial_sr_ssim'] = self.calculate_metrics(partial_sr_img, hr_img)
        
        # 记录处理时间
        metrics['bicubic_time'] = time_bicubic
        metrics['full_sr_time'] = time_full_sr
        metrics['partial_sr_time'] = time_partial_sr
        metrics['speedup'] = time_full_sr / time_partial_sr if time_partial_sr > 0 else 0
        metrics['total_time'] = time.time() - start_time_total
        
        # 获取重要区域边界框 (用于可视化)
        boxes, _ = self.sr_fusion.detect_important_regions(lr_path)
        
        # 8. 可视化结果（如果需要）
        if visualize:
            # 在原图上绘制边界框
            vis_lr_img = np.copy(lr_img)
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(vis_lr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 创建可视化图像
            plt.figure(figsize=(20, 15))
            
            # 原图和检测区域
            plt.subplot(2, 3, 1)
            plt.title("LR image with detection")
            plt.imshow(cv2.cvtColor(vis_lr_img, cv2.COLOR_BGR2RGB))
            
            # 掩码图
            if region_mask is not None:
                plt.subplot(2, 3, 2)
                plt.title("Critical area mask")
                plt.imshow(region_mask, cmap='jet')
            else:
                plt.subplot(2, 3, 2)
                plt.title("No critical areas detected")
                plt.axis('off')
            
            # 高分辨率参考图（如果有）
            if hr_img is not None:
                plt.subplot(2, 3, 3)
                plt.title("HR reference")
                plt.imshow(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
            else:
                plt.subplot(2, 3, 3)
                plt.title("No HR reference available")
                plt.axis('off')
            
            # 双三次上采样结果
            plt.subplot(2, 3, 4)
            if hr_img is not None:
                plt.title(f"Bicubic\nPSNR: {metrics['bicubic_psnr']:.2f}dB\nSSIM: {metrics['bicubic_ssim']:.4f}\nTime: {metrics['bicubic_time']:.3f}s")
            else:
                plt.title(f"Bicubic\nTime: {metrics['bicubic_time']:.3f}s")
            plt.imshow(cv2.cvtColor(bicubic_img, cv2.COLOR_BGR2RGB))
            
            # 整图超分结果
            plt.subplot(2, 3, 5)
            if hr_img is not None:
                plt.title(f"Full SR\nPSNR: {metrics['full_sr_psnr']:.2f}dB\nSSIM: {metrics['full_sr_ssim']:.4f}\nTime: {metrics['full_sr_time']:.3f}s")
            else:
                plt.title(f"Full SR\nTime: {metrics['full_sr_time']:.3f}s")
            plt.imshow(cv2.cvtColor(full_sr_img, cv2.COLOR_BGR2RGB))
            
            # 部分区域超分结果
            plt.subplot(2, 3, 6)
            if hr_img is not None:
                plt.title(f"Partial SR\nPSNR: {metrics['partial_sr_psnr']:.2f}dB\nSSIM: {metrics['partial_sr_ssim']:.4f}\nTime: {metrics['partial_sr_time']:.3f}s\nSpeedup: {metrics['speedup']:.2f}x")
            else:
                plt.title(f"Partial SR\nTime: {metrics['partial_sr_time']:.3f}s\nSpeedup: {metrics['speedup']:.2f}x")
            plt.imshow(cv2.cvtColor(partial_sr_img, cv2.COLOR_BGR2RGB))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_filename}_comparison.png"), dpi=300)
            plt.close()
        
        return metrics
    
    
    def run_batch_experiment(self, lr_dir, output_dir, limit=None, visualize=True):
        """对一批图像进行实验对比"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有低分辨率图像
        lr_files = [f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 限制处理的图像数量（如果指定）
        if limit and limit > 0:
            lr_files = lr_files[:limit]
        
        # 用于存储所有图像的指标
        all_metrics = []
        # 处理每张图像
        for idx, filename in enumerate(tqdm(lr_files, desc="Processing images")):
            lr_path = os.path.join(lr_dir, filename)
            img_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(img_output_dir, exist_ok=True)
            
            print(f"\n处理图像 {idx+1}/{len(lr_files)}: {filename}")
            metrics = self.run_single_image_experiment(lr_path, img_output_dir, visualize)
            metrics['filename'] = filename
            all_metrics.append(metrics)
            
            # 输出当前图像的指标
            if 'bicubic_psnr' in metrics:
                print(f"  双三次上采样: PSNR={metrics['bicubic_psnr']:.2f}dB, SSIM={metrics['bicubic_ssim']:.4f}, 时间={metrics['bicubic_time']:.3f}s")
                print(f"  整图超分: PSNR={metrics['full_sr_psnr']:.2f}dB, SSIM={metrics['full_sr_ssim']:.4f}, 时间={metrics['full_sr_time']:.3f}s")
                print(f"  区域超分+融合: PSNR={metrics['partial_sr_psnr']:.2f}dB, SSIM={metrics['partial_sr_ssim']:.4f}, 时间={metrics['partial_sr_time']:.3f}s")
            else:
                print(f"  双三次上采样: 时间={metrics['bicubic_time']:.3f}s")
                print(f"  整图超分: 时间={metrics['full_sr_time']:.3f}s")
                print(f"  区域超分+融合: 时间={metrics['partial_sr_time']:.3f}s")
            print(f"  加速比: {metrics['speedup']:.2f}x")
            

        
        # 生成汇总报告
        if all_metrics:
            self.generate_summary_report(all_metrics, output_dir)
        
        return all_metrics
    
    def generate_summary_report(self, all_metrics, output_dir):
        """生成汇总报告和可视化图表"""
        # 转换为DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # 计算平均值
        has_psnr_ssim = 'bicubic_psnr' in metrics_df.columns
        avg_metrics = metrics_df.mean(numeric_only=True)
        
        # 添加平均行
        avg_metrics_dict = avg_metrics.to_dict()
        avg_metrics_dict['filename'] = 'AVERAGE'
        all_metrics.append(avg_metrics_dict)
        
        # 更新DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # 保存到CSV
        csv_path = os.path.join(output_dir, 'metrics_summary.csv')
        metrics_df.to_csv(csv_path, index=False)
        
        # 打印平均指标
        print("\n=========== 实验总结 ===========")
        print(f"处理图像数量: {len(all_metrics) - 1}")  # 减去平均行
        if has_psnr_ssim:
            print(f"平均PSNR - 双三次上采样: {avg_metrics['bicubic_psnr']:.2f}dB")
            print(f"平均PSNR - 整图超分: {avg_metrics['full_sr_psnr']:.2f}dB")
            print(f"平均PSNR - 区域超分+融合: {avg_metrics['partial_sr_psnr']:.2f}dB")
            print(f"平均SSIM - 双三次上采样: {avg_metrics['bicubic_ssim']:.4f}")
            print(f"平均SSIM - 整图超分: {avg_metrics['full_sr_ssim']:.4f}")
            print(f"平均SSIM - 区域超分+融合: {avg_metrics['partial_sr_ssim']:.4f}")
        
        print(f"平均处理时间 - 双三次上采样: {avg_metrics['bicubic_time']:.3f}s")
        print(f"平均处理时间 - 整图超分: {avg_metrics['full_sr_time']:.3f}s")
        print(f"平均处理时间 - 区域超分+融合: {avg_metrics['partial_sr_time']:.3f}s")
        print(f"平均加速比: {avg_metrics['speedup']:.2f}x")
        print(f"详细结果已保存至: {csv_path}")
        
        # 绘制汇总图表
        plt.figure(figsize=(15, 10))
        
        if has_psnr_ssim:
            # PSNR对比图
            plt.subplot(2, 2, 1)
            plt.title("PSNR Comparison")
            bars1 = plt.bar(['Bicubic', 'Full SR', 'Partial SR'], 
                    [avg_metrics['bicubic_psnr'], avg_metrics['full_sr_psnr'], avg_metrics['partial_sr_psnr']])
            plt.ylabel('PSNR (dB)')
            # 在柱状图上添加数值标签
            for bar in bars1:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # SSIM对比图
            plt.subplot(2, 2, 2)
            plt.title("SSIM Comparison")
            bars2 = plt.bar(['Bicubic', 'Full SR', 'Partial SR'], 
                    [avg_metrics['bicubic_ssim'], avg_metrics['full_sr_ssim'], avg_metrics['partial_sr_ssim']])
            plt.ylabel('SSIM')
            # 在柱状图上添加数值标签
            for bar in bars2:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.4f}', ha='center', va='bottom')
        else:
            # 如果没有PSNR和SSIM指标，显示空白
            plt.subplot(2, 2, 1)
            plt.title("PSNR Comparison (No data)")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.title("SSIM Comparison (No data)")
            plt.axis('off')
        
        # 处理时间对比
        plt.subplot(2, 2, 3)
        plt.title("Processing Time")
        bars3 = plt.bar(['Bicubic', 'Full SR', 'Partial SR'], 
                [avg_metrics['bicubic_time'], avg_metrics['full_sr_time'], avg_metrics['partial_sr_time']])
        plt.ylabel('Time (s)')
        # 在柱状图上添加数值标签
        for bar in bars3:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}s', ha='center', va='bottom')
        
        # 加速比
        plt.subplot(2, 2, 4)
        plt.title(f"Speed Improvement: {avg_metrics['speedup']:.2f}x")
        speedup_data = [1, max(0, avg_metrics['speedup']-1)]
        plt.pie(speedup_data, 
                labels=['Base Speed', 'Improvement'], 
                autopct='%1.1f%%', 
                startangle=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300)
        plt.close()


if __name__ == "__main__":
    # 创建实验对象
    experiment = SRExperiment(
        sr_model_path="/root/EdgeSR/basemodels/basicSR/srformer/SRFormer_SRx4_DF2K.pth",
        model_type="SRFormer",
        scale=4
    )
    
    # 运行批量实验
    experiment.run_batch_experiment(
        lr_dir="/root/autodl-tmp/Datasets/Urban100/image_SRF_4/LR",
        output_dir="/root/autodl-tmp/output_edge_device_against/Urban100_SRF4@SRFormer",
        limit = None,  
        visualize=True
    )