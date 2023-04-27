import torch
import os
import argparse
import cv2
from matplotlib import pyplot as plt
from datasets.crowd import Crowd
import numpy as np
from models.fusion import fusion_model
from models.SCANet import SCANet
from models.SCANet_v2 import SCANet
from models.CSRNet import CSRNet
from utils.evaluation import eval_game, eval_relative


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='/home/sawyer/jupyter/CrowdCountingRGBT/BL+IADM for RGBT Crowd Counting/data/RGBT-CC-CVPR2021',  #DroneRGBT_wsy  # RGBT-CC-CVPR2021
                        help='training data directory')
parser.add_argument('--save-dir', default='/home/sawyer/jupyter/CrowdCountingRGBT/OneSCRNet/RGBTCC_pth/1107-103415',     # 1014-160206 for IADM;  1014-121731 for SCANet
                        help='model directory')
parser.add_argument('--model', default='best_model_5.pth'
                    , help='model name')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':


    datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    visual_dir = '/home/sawyer/jupyter/CrowdCountingRGBT/OneSCRNet/OneSCRNet_RGBTCC_1107_103415'
    if not os.path.isdir(visual_dir):
        os.makedirs(visual_dir)

    #model = fusion_model()
    #model = SCANet()
    model = CSRNet()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'

        img_name = name
        img_name = (str(img_name[0]))[0:4]
        #print(target.shape)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error

            #wsy 可视化s
            #outputs = outputs.data.cpu().numpy()
            output_vis = outputs[0][0].cpu().numpy()
            target_vis = target[0].cpu().numpy()
            out_count = np.sum(output_vis)
            gt_count = np.sum(target_vis)
            H, W = output_vis.shape
            ratio = H / target_vis.shape[0]
            target_vis = cv2.resize(target_vis, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio * ratio)
            #cv2.imwrite(os.path.join(visual_dir, img_name + '.jpg'), output_vis * 255)
            plt.imsave(os.path.join(visual_dir, img_name+'_'+str(gt_count)+'_'+str(out_count) + '.jpg'), output_vis, cmap = 'magma')

    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f} Re {relative:.4f}, '.\
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

    print(log_str)

