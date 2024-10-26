import torch
from utils import fast_hist, per_class_iou
import numpy as np
def train(model, optimizer, dataloader,loss_fn,device, is_gta5 = False):
    model.train()
    hist = np.zeros((19, 19))
    values = []
    for batch_idx, (inputs,targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device).long()
        outputs = model(inputs)
        if is_gta5: 
          targets[targets > 18] = 255
        loss = loss_fn(outputs[0],targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = outputs[0].max(1)
        

        hist += fast_hist(targets.cpu().flatten().numpy(), predicted.cpu().flatten().numpy(), 19)
    print(per_class_iou(hist))
    miou = np.mean(per_class_iou(hist))
    print({"Final mIoU:": miou})

def val(model, dataloader,device):
    model.eval()

    hist = np.zeros((19, 19))
    with torch.no_grad():
        for batch_idx,(inputs,targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            hist += fast_hist(targets.cpu().flatten().numpy(), predicted.cpu().flatten().numpy(), 19)
        miou = np.mean(per_class_iou(hist))
        print({"Final mIoU:": miou})
        return per_class_iou(hist)