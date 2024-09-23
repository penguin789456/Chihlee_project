import torch
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2

# 加載檢查點
checkpoint = torch.load("model_final.pth", map_location=torch.device('cpu'))

# 保存模型權重到一個新文件（僅保存模型部分）
torch.save(checkpoint['model'], "model_weights.pth")

# 設定配置
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 設置類別數
cfg.MODEL.WEIGHTS = "model_final.pth"  # 使用保存的檢查點文件
cfg.MODEL.DEVICE = 'cpu'  # 使用 CPU

# 檢查是否創建輸出目錄
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 建立預測器
predictor = DefaultPredictor(cfg)

# 用於測試的圖像
image_path = "PNGTEMP/w644.png"  # 請確保該路徑是正確的
im = cv2.imread(image_path)

# 檢查圖像是否加載正確
if im is None:
    raise ValueError(f"Image at path {image_path} could not be loaded. Please check the file path.")

# 進行推理
outputs = predictor(im)
print(outputs)
