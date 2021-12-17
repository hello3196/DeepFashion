from src.dataset import DeepFashionCAPDataset
from src.const import base_path
import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from src import const
from src.utils import parse_args_and_merge_const
from tensorboardX import SummaryWriter
import os
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import normalize
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io, transform

def rescale(image, output_size):
    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h>w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(image, (new_h, new_w), mode='constant')
    return img

def center_crop(image, output_size):
    output_size = (output_size, output_size)
    h, w = image.shape[:2]
    new_h, new_w = output_size
    top = int((h-new_h) / 2)
    left = int((w-new_w) / 2)
    image = image[top: top+new_h, left: left + new_w]
    return image

if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(base_path + const.USE_CSV)
    net = const.USE_NET()
    net = net.to(const.device)
    
    net.load_state_dict(torch.load('models/whole.pkl'))

    print('Now Evaluate..')
    with torch.no_grad():
        net.eval()
        evaluator = const.EVALUATOR()
        sample = [0] * 8
        sample[0] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/1967/1967JetBlackJeansWideStraight.jpg'), 224), 224))
        sample[1] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/A-line/ALineDenimSkirtBlackGray.jpg'), 224), 224))
        sample[2] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/Button/ButtonKnitOnePieceBlack.jpg'), 224), 224))
        sample[3] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/Down/DownPufferShortBubbleJacketLightGray.jpg'), 224), 224))
        sample[4] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/Fleece/FleeceZipUpJacketIvory.jpg'), 224), 224))
        sample[5] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/Minimal/MinimalStandardCargoJoggerPants.jpg'), 224), 224))
        sample[6] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/tag/TagToHoodieBlack.jpg'), 224), 224))
        sample[7] = to_tensor(center_crop(rescale(io.imread(base_path + 'real/Wool/WoolRichTrenchMackintoshCoatBlack.jpg'), 224), 224))
        sample = torch.stack(sample)
        sample = normalize(sample, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).float().to(const.device)
        tmg = sample
        sample = {"image": sample}
        for c in range(8):
            temp = normalize(tmg[c], mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                      std=[1/0.229, 1/0.224, 1/0.225])
            temp = to_pil_image(temp.squeeze())
            temp.save(f"img{c}.jpg")
        output = net(sample)
        atm = torch.sum(output['attention_map'], dim=1, keepdim=True)
        atm = F.interpolate(atm, scale_factor=8, mode='bilinear')
        lm = torch.sum(output['lm_pos_map'], dim=1, keepdim=True)
        atm = atm.cpu().numpy()
        lm = lm.cpu().numpy()
        for c in range(8):
            temp = sns.heatmap(atm[c].squeeze())
            plt.savefig(f'att_map{c}.png')
            plt.close()
            temp = sns.heatmap(lm[c].squeeze())
            plt.savefig(f'lm{c}.png')
            plt.close()

