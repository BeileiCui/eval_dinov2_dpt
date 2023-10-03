"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse
import numpy as np
import sys
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, '/mnt/data-hdd2/Beilei/Repository/DPT')
import util.io

import torch.utils.data as data
from torchvision.transforms import Compose
import torch.nn.functional as F
from PIL import Image
import matplotlib

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from mytest.scared_dataset import SCAREDDataset

#from util.misc import visualize_attention

def compute_CIs(sample, alpha=0.05):
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)

    dof = len(sample) - 1 
    
    interval = stats.t.interval(alpha, dof, loc=mean, scale=std / np.sqrt(len(sample)))
    
    return interval

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

class BadPixelMetric:
    def __init__(self, threshold=1.25, depth_cap=150):
        self.__threshold = threshold
        self.__depth_cap = depth_cap

    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):
        # transform predicted disparity to aligned depth
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]

        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediciton_depth = 1.0 / prediction_aligned

        # bad pixel
        err = torch.zeros_like(prediciton_depth, dtype=torch.float)

        err[mask == 1] = torch.max(
            prediciton_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediciton_depth[mask == 1],
        )

        err[mask == 1] = (err[mask == 1] > self.__threshold).float()

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p), prediciton_depth
    
def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

def run(input_path, output_path, model_path, model_type="dpt_scared", optimize=True, eval=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    
    # load network
    if model_type == "dpt_scared":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            scale=1,
            shift=0,
            invert=False,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    elif model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.0006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_scared|pt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"
        
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)
    
    ds = SCAREDDataset("test", input_path, target_W = 384, target_H = 384)
    dl = data.DataLoader(ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)
    
    metric = BadPixelMetric(depth_cap=150)
    
    img_shape = [1024,1280]
    
    if eval:
        num_samples = len(ds)

        silog = np.zeros(num_samples, np.float32)
        log10 = np.zeros(num_samples, np.float32)
        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1 = np.zeros(num_samples, np.float32)
        d2 = np.zeros(num_samples, np.float32)
        d3 = np.zeros(num_samples, np.float32)
        
    # create output folder
    pred_output_path = os.path.join(output_path,"pred_depth")
    vis_output_path = os.path.join(output_path,"vis_depth")
    os.makedirs(pred_output_path, exist_ok=True)
    os.makedirs(vis_output_path, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl)):
            # print("  processing ({}/{})".format(i + 1, len(ds)))
            
            # to device
            img = np.array(batch["image"].squeeze(0))
            depth = np.array(batch["depth"].squeeze(0))
            # depth[depth <=0] = depth.max()
            
            img_input = transform({"image": img})["image"]

            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            # run model
            prediction = model.forward(sample)
            
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img_shape,
                mode="bilinear",
                align_corners=False,
            )
            prediction = prediction.squeeze().cpu().numpy()
            
            mask = np.zeros_like(depth)
            mask[depth > 0] = 1

            loss, aligned_prediciton_depth = metric(torch.from_numpy(prediction).unsqueeze(0), torch.from_numpy(depth).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
            aligned_prediciton_depth = aligned_prediciton_depth.squeeze().cpu().numpy()
            
            aligned_prediciton_depth[aligned_prediciton_depth < 0 ] = 0
            aligned_prediciton_depth[aligned_prediciton_depth > 150 ] = 150
            
            if eval:
                valid_mask = np.array(mask, dtype=bool)
                silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(depth[valid_mask], aligned_prediciton_depth[valid_mask])
            vis_pred = render_depth(aligned_prediciton_depth)
            
            pred_file_name = os.path.join(pred_output_path, batch["sequence"][0] + "_" +  batch["keyframe"][0] + "_" + batch["frame_id"][0] + ".tiff")
            vis_file_name = os.path.join(vis_output_path, batch["sequence"][0] + "_" +  batch["keyframe"][0] + "_" + batch["frame_id"][0] + ".png")
            
            
            
            
            cv2.imwrite(pred_file_name,aligned_prediciton_depth)
            vis_pred.save(vis_file_name)
    
    if eval:
        
        CI_abs_rel = compute_CIs(abs_rel)
        CI_sq_rel = compute_CIs(sq_rel)
        CI_rms = compute_CIs(rms)
        CI_log_rms = compute_CIs(log_rms)
        CI_d1 = compute_CIs(d1)
        
        print("{:>7},       95% CIs,     {:>7},      95% CIs,     {:>7},      95% CIs,     {:>7},      95% CIs,     {:>7}, {:>7}, {:>7},         95% CIs,    {:>7}, {:>7}".format(
            'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10', 'd1', 'd2', 'd3'))
        print("{:7.3f}, [{:7.3f},{:7.3f}], {:7.3f}, [{:7.3f},{:7.3f}], {:7.3f}, [{:7.3f},{:7.3f}], {:7.3f}, [{:7.3f},{:7.3f}], {:7.3f}, {:7.3f}, {:7.3f}, [{:7.3f},{:7.3f}], {:7.3f}, {:7.3f}".format(
            abs_rel.mean(), CI_abs_rel[0], CI_abs_rel[1], sq_rel.mean(), CI_sq_rel[0], CI_sq_rel[1], 
            rms.mean(), CI_rms[0], CI_rms[1], log_rms.mean(), CI_log_rms[0], CI_log_rms[1], 
            silog.mean(), log10.mean(), 
            d1.mean(), CI_d1[0], CI_d1[1], d2.mean(), d3.mean()))
        
    print("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="/mnt/data-hdd2/Beilei/Dataset/SCARED", help="folder with input images"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/mnt/data-hdd2/Beilei/Repository/DPT/mytest/output",
        help="folder for output images",
    )
    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_scared",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.add_argument("--eval", default=True, help="evaluate model at the same time") 
    
    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_scared": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
        args.eval,
    )
