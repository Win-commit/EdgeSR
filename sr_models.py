import cv2
import glob
import numpy as np
import os
import torch
import sys
sys.path.append('basemodels/DRCT')
sys.path.append('basemodels/basicSR')
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.edsr_arch import EDSR
from basicsr.utils.registry import ARCH_REGISTRY 

class BaseSRModel:
    def inference(self, img_path):
        """输入图片路径，返回最终结果（模型内部自行处理预处理、推理、后处理）"""
        raise NotImplementedError



class DRCTModel(BaseSRModel):
    def __init__(self, model_path=None, device=None, scale=4, tile_size=None, tile_overlap=32, window_size=16):
        self.model_path = model_path
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.window_size = window_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = self._load_model()

    def _load_model(self):
        from drct.archs.DRCT_arch import DRCT
        model = DRCT(upscale=self.scale, in_chans=3, img_size=64, window_size=self.window_size, 
                     compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, 
                     img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6, 
                     mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        if self.model_path is not None:
            model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
        model.eval()
        return model.to(self.device)
    
    
    def inference(self, img_path):
        # 预处理
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
        w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
        original_size = (h_old, w_old)
        # 推理
        if self.tile_size is None:
            output = self.model(img)
        else:
            b, c, h, w = img.size()
            tile = min(self.tile_size, h, w)
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
        # 后处理
        h_old, w_old = original_size
        output = output[..., :h_old * self.scale, :w_old * self.scale]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output

class BasicSRModel(BaseSRModel):
    def __init__(self, model_path=None, model_name=None, device=None, scale=4):
        '''
        model_name:[EDSR,ESRGAN,HAT,HMA,CAT,SRFormer]
        '''
        self.model_path = model_path
        self.model_name = model_name
        self.scale = scale
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if model_name == 'ESRGAN':
            if scale != 4:
                raise ValueError(f"ESRGAN 只支持放大4倍")
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            self.model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.window_size = None
        elif model_name == 'EDSR':
            self.model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=scale)
            self.model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.window_size = None
        elif model_name == 'HAT':
            if scale != 4:
                raise ValueError(f"HAT 只支持放大4倍")
            import hat.archs # 触发basicsr的注册机制
            self.model = ARCH_REGISTRY.get('HAT')(
                upscale=4,
                window_size = 16,
                upsampler='pixelshuffle',
                depths = (6,6,6,6,6,6),
                embed_dim = 180,
                num_heads = (6,6,6,6,6,6),
                mlp_ratio = 2,
            )
            self.model.load_state_dict(torch.load(self.model_path)['params_ema'], strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.window_size = 16
        elif model_name == 'HMA':
            if scale != 4:
                raise ValueError(f"HMA 只支持放大4倍")
            import hma.archs  
            self.model = ARCH_REGISTRY.get('HMANet')(
                upscale=4,
                in_chans = 3,
                img_size = 64,
                interval_size = 4,
                window_size = 16,
                upsampler='pixelshuffle',
                depths = (6,6,6,6,6,6),
                embed_dim = 180,
                num_heads = (6,6,6,6,6,6),
                mlp_ratio = 2,
                resi_connection = '1conv',
            )
            self.model.load_state_dict(torch.load(self.model_path)['params_ema'], strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.window_size = 16
        elif model_name == 'CAT':
            if scale != 4:
                raise ValueError(f"CAT 只支持放大4倍")
            import cat.archs
            self.model = ARCH_REGISTRY.get('CAT')(
                upscale=4,
                in_chans = 3,
                img_size = 64,
                split_size_0 = [4,4,4,4,4,4],
                split_size_1 = [0,0,0,0,0,0],
                img_range = 1,
                depth = [6,6,6,6,6,6],
                embed_dim = 180,
                num_heads = [6,6,6,6,6,6],
                mlp_ratio = 4,
                resi_connection = '1conv',
                block_name = 'CATB_axial',
                upsampler = 'pixelshuffle',
                )
            self.model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.window_size = None
        elif model_name == 'SRFormer':
            import srformer.archs
            self.model = ARCH_REGISTRY.get('SRFormer')(
                upscale=4,
                in_chans = 3,
                img_size = 64,
                window_size = 22,
                img_range = 1.0,
                depths = [6,6,6,6,6,6],
                embed_dim = 180,
                num_heads = [6,6,6,6,6,6],
                mlp_ratio = 2,
                upsampler = 'pixelshuffle',
                resi_connection = '1conv',
            )
            self.model.load_state_dict(torch.load(self.model_path)['params'], strict=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.window_size = 22
            
    def inference(self, img_path):
        if self.window_size is None and self.model_name != 'CAT':
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(img)
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
        elif self.model_name == 'CAT':
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(self.device)
            _, _, h_old, w_old = img.size()
            
            # 使用split_size进行padding
            split_size = 4  # 从split_size_0参数获取
            h_pad = (h_old // split_size + 1) * split_size - h_old
            w_pad = (w_old // split_size + 1) * split_size - w_old
            
            # padding处理
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
            original_size = (h_old, w_old)
            
            with torch.no_grad():
                output = self.model(img)
            # 恢复原始尺寸
            output = output[..., :h_old * self.scale, :w_old * self.scale]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(self.device)
            _, _, h_old, w_old = img.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
            original_size = (h_old, w_old)
            with torch.no_grad():
                output = self.model(img)
            h_old, w_old = original_size
            output = output[..., :h_old * self.scale, :w_old * self.scale]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
        return output

class SuperResolutionModel:
    """统一入口，只保留super_resolve和batch_process，所有细节交给模型类"""
    def __init__(self, model_type='DRCT', **kwargs):
        if model_type == 'DRCT':
            self.model = DRCTModel(**kwargs)
        else:
            self.model = BasicSRModel(model_name=model_type, **kwargs)

        self.scale = self.model.scale  # 便于命名输出文件
    def super_resolve(self, img_path, save_path=None):
        result = self.model.inference(img_path)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, result)
        return result
    
    def batch_process(self, input_dir, output_dir, name_suffix='_SR'):
        os.makedirs(output_dir, exist_ok=True)
        for idx, path in enumerate(sorted(glob.glob(os.path.join(input_dir, '*')))):
            if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            imgname = os.path.splitext(os.path.basename(path))[0]
            print(f'处理第 {idx} 张图片: {imgname}')
            output_path = os.path.join(output_dir, f'{imgname}{name_suffix}_X{self.scale}.png')
            self.super_resolve(path, output_path)