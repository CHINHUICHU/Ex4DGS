"""
Render and evaluate an edited Ex4DGS model.

Loads the edited static PLY + original dynamic PLY, renders test cameras,
and computes PSNR/SSIM/LPIPS/CLIP vs original ground-truth frames.
"""
import os
import re
import json
import argparse

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

import clip
from skimage.metrics import structural_similarity as sk_ssim
from lpipsPyTorch import lpips

from scene import Scene
from scene.c_gaussian_model import CGaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


# ── CLIP setup ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


def clip_similarity(img_tensor: torch.Tensor, text: str) -> float:
    pil = T.ToPILImage()(img_tensor.detach().cpu().clamp(0, 1))
    img_in = clip_preprocess(pil).unsqueeze(0).to(device)
    txt_in = clip.tokenize([text]).to(device)
    with torch.no_grad():
        i_feat = clip_model.encode_image(img_in)
        t_feat = clip_model.encode_text(txt_in)
    i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
    t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
    return (i_feat * t_feat).sum().item()


def evaluate(dataset, opt, pipe, ply_path, prompt, save_dir=None, mode=0):
    """
    mode=0: render static+dynamic (full scene)
    mode=1: render static only
    """
    gaussians = CGaussianModel(
        sh_degree=dataset.sh_degree,
        duration=dataset.duration,
        interval=dataset.time_interval,
        time_pad=dataset.time_pad,
        interp_type=dataset.interp_type,
        rot_interp_type=dataset.rot_interp_type,
        time_pad_type=dataset.time_pad_type,
        var_pad=dataset.var_pad,
        kernel_size=dataset.kernel_size,
    )

    # Load cameras without loading PLY through Scene
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)

    # Override with edited PLY
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    test_cams, test_imgs = scene.getTestCameras(shuffle=False, return_as="list")
    test_gt = list(test_imgs)

    if save_dir:
        os.makedirs(os.path.join(save_dir, "renders"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "gt"), exist_ok=True)

    psnrs, ssims, lpipss, clip_scores = [], [], [], []

    for idx, (cam, gt) in enumerate(tqdm(zip(test_cams, test_gt), total=len(test_cams), desc="Evaluating")):
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg,
                         near=dataset.near, far=dataset.far, mode=mode)
            rendered = pkg["render"]

            gt = gt.cuda()
            if gt.shape != rendered.shape:
                gt = F.interpolate(gt.unsqueeze(0), size=rendered.shape[-2:],
                                   mode="bilinear", align_corners=False).squeeze(0)

            psnrs.append(psnr(rendered.unsqueeze(0), gt.unsqueeze(0)))
            ssims.append(ssim(rendered.unsqueeze(0), gt.unsqueeze(0)))
            lpipss.append(lpips(rendered.unsqueeze(0), gt.unsqueeze(0), net_type="alex"))
            clip_scores.append(clip_similarity(rendered, prompt))

            if save_dir:
                torchvision.utils.save_image(rendered, os.path.join(save_dir, "renders", f"{idx:04d}.png"))
                torchvision.utils.save_image(gt,       os.path.join(save_dir, "gt",      f"{idx:04d}.png"))

    avg = {
        "PSNR":  float(torch.stack(psnrs).mean()),
        "SSIM":  float(torch.stack(ssims).mean()),
        "LPIPS": float(torch.stack(lpipss).mean()),
        "CLIP":  float(np.mean(clip_scores)),
    }

    print("\n=== Results ===")
    print(f"  Prompt : {prompt}")
    print(f"  PLY    : {ply_path}")
    print(f"  Frames : {len(psnrs)}")
    print(f"  PSNR   : {avg['PSNR']:.4f}")
    print(f"  SSIM   : {avg['SSIM']:.4f}")
    print(f"  LPIPS  : {avg['LPIPS']:.4f}")
    print(f"  CLIP   : {avg['CLIP']:.4f}")

    if save_dir:
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(avg, f, indent=2)
        print(f"  Saved  : {save_dir}")

    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    opt   = OptimizationParams(parser)
    pipe  = PipelineParams(parser)
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to edited static PLY (dynamic PLY must be alongside it)")
    parser.add_argument("--prompt",   type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--mode",     type=int, default=0,
                        help="Render mode: 0=full, 1=static-only")
    parser.add_argument("--quiet",    action="store_true")
    args = get_combined_args(parser)
    safe_state(silent=True)

    m = model.extract(args)
    evaluate(
        m, opt.extract(args), pipe.extract(args),
        args.ply_path, args.prompt,
        getattr(args, "save_dir", None),
        getattr(args, "mode", 0),
    )
