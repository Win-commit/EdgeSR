from openVS import PatchImages
import SpLiCE.splice as splice
import torch
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_Important_concepts(image_path, splicemodel, preprocess, vocabulary, topk: int = 5):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")
    sparse_weights = splicemodel.encode_image(image) 
    sparse_weights = sparse_weights.squeeze()
    topk_concepts = {}
    for weight_idx in torch.sort(sparse_weights, descending=True)[1][:topk]:
        topk_concepts[vocabulary[weight_idx]] = sparse_weights[weight_idx]

    return topk_concepts


class Incorporate:
    def __init__(self):
        self.results = {}

    def compute_metrics(self, box_a, box_b):
        a_x1, a_y1, a_x2, a_y2 = box_a
        b_x1, b_y1, b_x2, b_y2 = box_b

        # 计算交集区域
        inter_x1 = max(a_x1, b_x1)
        inter_y1 = max(a_y1, b_y1)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        # 如果没有交集，返回0
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0, 0.0, 0.0

        # 计算交集面积
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # 计算框A和框B的面积
        area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
        area_b = (b_x2 - b_x1) * (b_y2 - b_y1)

        # 计算IoU
        union_area = area_a + area_b - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0

        # 计算包含率 - A包含在B中的比例和B包含在A中的比例
        contain_a_in_b = inter_area / area_a if area_a > 0 else 0.0
        contain_b_in_a = inter_area / area_b if area_b > 0 else 0.0

        return iou, contain_a_in_b, contain_b_in_a

    def should_merge(self, box_a, box_b, iou_threshold, containment_threshold):
        """判断两个框是否应该合并"""
        iou, contain_a_in_b, contain_b_in_a = self.compute_metrics(box_a, box_b)
        
        # 如果IoU超过阈值或者一个框大部分包含在另一个框中，则合并
        return (iou >= iou_threshold or 
                contain_a_in_b >= containment_threshold or 
                contain_b_in_a >= containment_threshold)

    def merge_boxes(self, boxes):
        """合并一组框，返回单个合并框"""
        if not boxes:
            return None
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        return [x1, y1, x2, y2]

    def __call__(self, crop_pixels, IoU=0.75,containment=0.7):
        # 如果没有框，直接返回
        if not crop_pixels:
            self.results = {}
            return
        
        # 将所有框和键收集到一个列表中
        all_boxes_with_keys = []
        for key, boxes in crop_pixels.items():
            for box in boxes:
                all_boxes_with_keys.append((box, key))
        
        # 持续合并框，直到没有更多变化
        changed = True
        while changed:
            changed = False
            n = len(all_boxes_with_keys)
            merged = [False] * n
            new_all_boxes_with_keys = []
            
            for i in range(n):
                if merged[i]:
                    continue  # 如果框已经被合并，跳过
                
                box_i, key_i = all_boxes_with_keys[i]
                current_boxes = [box_i]
                current_keys = {key_i}
                merged[i] = True
                
                # 查找所有与当前框重叠的框
                for j in range(i+1, n):
                    if merged[j]:
                        continue
                    
                    box_j, key_j = all_boxes_with_keys[j]
                    # 使用当前合并的框进行IoU计算
                    merged_box = self.merge_boxes(current_boxes)
                    if self.should_merge(merged_box, box_j, iou_threshold=IoU, containment_threshold=containment):
                        current_boxes.append(box_j)
                        current_keys.add(key_j)
                        merged[j] = True
                        changed = True  # 标记有变化发生
                
                # 添加合并后的框到新列表
                merged_box = self.merge_boxes(current_boxes)
                merged_key = ','.join(sorted(current_keys))
                new_all_boxes_with_keys.append((merged_box, merged_key))
            
            # 更新框列表
            all_boxes_with_keys = new_all_boxes_with_keys
        
        # 将结果保存到字典
        results = {}
        for box, key in all_boxes_with_keys:
            if key not in results:
                results[key] = []
            results[key].append(box)
        
        self.results = results


def get_patches(concepts:list, image_path, cropper: PatchImages, IoU: float = 0.75, containment: float = 0.7, BOX_TRESHOLD: float = 0.3, TEXT_TRESHOLD: float = 0.25):
    '''
    Output:object1,object2...:[box1...]
    '''
    cropper.set_concepts(concepts)
    _, crop_pixels = cropper(image_path, BOX_TRESHOLD, TEXT_TRESHOLD)
    conform = Incorporate()
    conform(crop_pixels,IoU,containment)

    return conform.results


if __name__ == '__main__':
    folder_path = "/root/autodl-tmp/Datasets/DIV2K_valid_LR_bicubic/X4"
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files_with_path = [os.path.join(folder_path, f) for f in png_files]
    results = []
    cropper = PatchImages(concepts = [])
    splicemodel = splice.load("open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000, l1_penalty=0.15, return_weights=True, device="cuda")
    preprocess = splice.get_preprocess("open_clip:ViT-B-32")
    vocabulary = splice.get_vocabulary("laion", 10000) 
    
    for image_path in tqdm(png_files_with_path):
        num_patch = 0
        concepts = get_Important_concepts(image_path, splicemodel, preprocess, vocabulary, 5)
        concepts_ls = [concept for concept in concepts]
        patches = get_patches(concepts_ls, image_path, cropper)
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)


        # 遍历字典，绘制边界框
        for label, boxes in patches.items():
            for box in boxes:
                num_patch += 1
                x1, y1, x2, y2 = box
                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], width=2)
                # 添加标签文本
                draw.text((x1, y1 - 10), label)

        # 保存标注后的图片
        file_name = os.path.basename(image_path)
        output_path = f'output/{file_name}'
        image.save(output_path)
        results.append(num_patch)
    
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=range(min(results), max(results) + 2), color='skyblue', edgecolor='black')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('patch_frequency_distribution.png', dpi=300, bbox_inches='tight')



