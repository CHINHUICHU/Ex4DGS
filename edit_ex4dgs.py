"""
Adapt Instruct-4DGS's edit_3d step for Ex4DGS's CGaussianModel.

Freezes the dynamic pool (_xyz_motion, _rotation_motion, etc.) and
optimizes only the static pool to match IP2P-edited t=0 images.
"""
import os
import sys
import re
import argparse

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from scene.c_gaussian_model import CGaussianModel
from scene import Scene
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


def load_edited_image(edited_dir, prompt_word, cam_idx, resolution):
    path = os.path.join(edited_dir, f"edited_{prompt_word}_original_time0_{cam_idx}.png")
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution[0], resolution[1]), Image.LANCZOS)
    return T.ToTensor()(img).cuda()


def get_prompt_word(prompt):
    word = prompt.strip().split()[-1]
    return re.sub(r'[^\w]', '', word).lower()


def edit_ex4dgs(dataset, opt, pipe, model_path, source_path, prompt, iterations=1000, save_path=None):
    prompt_word = get_prompt_word(prompt)
    edited_dir = os.path.join(source_path, prompt_word)
    print(f"Prompt word: '{prompt_word}', edited images dir: {edited_dir}")

    # Load pretrained Ex4DGS model
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
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    gaussians.spatial_lr_scale = scene.cameras_extent

    # Collect t=0 training cameras and their edited images
    # getTrainCameras() returns (cams_list, img_generator) in Ex4DGS
    train_cams_all, _ = scene.getTrainCameras(shuffle=False, get_img=False)
    t0_cams = [c for c in train_cams_all if c.timestamp == 0]

    cam_image_pairs = []
    for cam in t0_cams:
        # Extract camera index from image path (e.g. .../cam05/images/0000.png -> 5)
        match = re.search(r'cam(\d+)', cam.image_path)
        if match is None:
            continue
        cam_idx = int(match.group(1))
        resolution = (cam.image_width, cam.image_height)
        gt_edited = load_edited_image(edited_dir, prompt_word, cam_idx, resolution)
        if gt_edited is None:
            print(f"  Warning: no edited image for cam{cam_idx:02d}, skipping")
            continue
        cam_image_pairs.append((cam, gt_edited))

    print(f"Using {len(cam_image_pairs)} camera views for editing")

    # Static-only optimizer — freeze dynamic pool
    position_lr  = 1.6e-4 * scene.cameras_extent
    l = [
        {'params': [gaussians._xyz],          'lr': position_lr,  'name': 'xyz'},
        {'params': [gaussians._features_dc],   'lr': 2.5e-3,       'name': 'f_dc'},
        {'params': [gaussians._features_rest], 'lr': 2.5e-3/20,    'name': 'f_rest'},
        {'params': [gaussians._opacity],       'lr': 0.05,         'name': 'opacity'},
        {'params': [gaussians._scaling],       'lr': 5e-3,         'name': 'scaling'},
        {'params': [gaussians._rotation],      'lr': 1e-4,         'name': 'rotation'},
    ]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    progress = tqdm(range(iterations), desc="Editing static Gaussians")
    for iteration in progress:
        # Cycle through cameras
        cam, gt_img = cam_image_pairs[iteration % len(cam_image_pairs)]

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        render_pkg = render(cam, gaussians, pipe, bg, timestamp=0,
                            near=dataset.near, far=dataset.far, mode=1)  # mode=1 = static only
        image = render_pkg["render"]

        Ll1 = l1_loss(image, gt_img)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_img))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if iteration % 50 == 0:
            progress.set_postfix({"loss": f"{loss.item():.5f}"})

    # Save modified static PLY (dynamic PLY unchanged)
    if save_path is None:
        save_path = os.path.join(model_path, "point_cloud_edit", prompt, "point_cloud.ply")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gaussians.save_ply(save_path)
    print(f"Saved edited model to: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model   = ModelParams(parser, sentinel=True)
    opt     = OptimizationParams(parser)
    pipe    = PipelineParams(parser)
    parser.add_argument("--prompt",      type=str, required=True)
    parser.add_argument("--edit_iters",  type=int, default=1000)
    parser.add_argument("--edit_save",   type=str, default=None)
    parser.add_argument("--quiet",       action="store_true")
    args = get_combined_args(parser)

    safe_state(silent=True)
    m = model.extract(args)
    edit_ex4dgs(
        m, opt.extract(args), pipe.extract(args),
        args.model_path, m.source_path, args.prompt,
        args.edit_iters, getattr(args, 'edit_save', None),
    )
