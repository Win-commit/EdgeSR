import os
from groundingdino.util.inference import load_model, load_image, predict,annotate
import torch
from torchvision.ops import box_convert
from collections import defaultdict
import cv2

class PatchImages :
    def __init__(self, concepts: list, groundingDinoConfigPath = None, WeightsPath = None):
        self.concepts = concepts
        if groundingDinoConfigPath is None:
            CONFIG_PATH = os.path.join(os.getcwd(), "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        else:
            CONFIG_PATH = os.path.join(os.getcwd(),groundingDinoConfigPath)

        if WeightsPath is None:
            WEIGHTS_PATH = os.path.join(os.getcwd(), "groundingdino", "weights", "groundingdino_swint_ogc.pth")
        else:
            WEIGHTS_PATH = os.path.join(os.getcwd(), WeightsPath)

        self.model = load_model(CONFIG_PATH, WEIGHTS_PATH)
        self.saveImgs = defaultdict(list)

    def set_concepts(self, concepts: list):
        self.concepts = concepts
        
    def __call__(self, image_path: str, BOX_TRESHOLD: float = 0.3, TEXT_TRESHOLD: float = 0.25)->dict:
        '''
        Input: Picture, list of geo-objects(string)
        Output: list of image patches
        '''
        image_patches = {}
        crop_pixels = {}
        for geo_object in self.concepts:
            TEXT_PROMPT = geo_object
            image_patches[geo_object] = []
            crop_pixels[geo_object] = []
            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                                model=self.model, 
                                image=image, 
                                caption=TEXT_PROMPT, 
                                box_threshold=BOX_TRESHOLD, 
                                text_threshold=TEXT_TRESHOLD
                            )
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            self.saveImgs[geo_object].append(annotated_frame)
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                if abs(x2-x1) < 224 and abs(y2-y1) < 224:
                    continue
                image_patches[geo_object].append(image_source[int(y1):int(y2), int(x1):int(x2)])
                crop_pixels[geo_object].append([x1,y1,x2,y2])

        return image_patches,crop_pixels

    def save_annotation(self,output_path:str):
        '''
        Input: list of image patches: category->list of image patches
        Output: save the patches
        '''
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        for geo_object in self.concepts:
            for i, img in enumerate(self.saveImgs[geo_object]):
                cv2.imwrite(os.path.join(output_path, f"{geo_object}_{i}.jpg"), img)