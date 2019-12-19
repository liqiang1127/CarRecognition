import torch
from net import CarRecognitionNet

checkpoint = torch.load("./checkpoint/BEST_OURS.tar")
model = checkpoint['model']
torch.save(model.state_dict(), "./model/final_model.tar")
print("保存模型完成")
