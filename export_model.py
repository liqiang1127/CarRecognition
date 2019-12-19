import torch


def ext_model():
    checkpoint = torch.load("./checkpoint/BEST_OURS.tar")
    model = checkpoint['model']
    torch.save(model.state_dict(), "./model/final_model.tar")
    print("====以将最佳模型保存到model文件夹中====")
