"""
SDS refinement for Ex4DGS edited models.

Adapts Instruct-4DGS's refine_sds.py for CGaussianModel:
  - Loads the step-3 edited static PLY + original dynamic PLY
  - Freezes dynamic pool; applies IP2P-SDS gradient to static pool across all timestamps
  - Saves refined static PLY

Usage:
    cd /home/chinhui/Instruct-4DGS/Ex4DGS
    source /home/chinhui/miniforge3/bin/activate Ex4DGS
    python refine_sds_ex4dgs.py \
        --model_path output/ \
        --source_path /home/chinhui/Instruct-4DGS/data/dynerf/cook_spinach \
        --loader dynerf \
        --ply_path "output/point_cloud_edit/Make it look like a fauvism painting/point_cloud.ply" \
        --prompt "Make it look like a fauvism painting" \
        --guidance_scale 10.5 --image_guidance_scale 1.2 \
        --sds_iters 800 --resize 512
"""
import os
import sys
import re
import random
import math
import argparse

# Ex4DGS modules must take priority; ip2p_models lives in Instruct-4DGS root
_ex4dgs_dir = os.path.dirname(os.path.abspath(__file__))
_i4dgs_dir  = os.path.join(_ex4dgs_dir, '..')
sys.path.insert(0, _ex4dgs_dir)   # Ex4DGS scene/renderer/etc take priority
sys.path.insert(1, _i4dgs_dir)    # Instruct-4DGS provides ip2p_models only

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from einops import rearrange

from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

from ip2p_models.models.ip2p_pipeline import InstructPix2PixPipeline
from ip2p_models.models.ip2p_unet import UNet3DConditionModel

from scene import Scene
from scene.c_gaussian_model import CGaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args


# ── VAE encode helpers (same as refine_sds.py) ────────────────────────────────

def encode_1(ip2p, x):
    """Differentiable VAE encode (sample from latent dist)."""
    return ip2p.vae.encode(2 * x - 1).latent_dist.sample() * 0.18215

def encode_2(ip2p, x):
    """VAE encode mode (no gradient needed)."""
    return ip2p.vae.encode(2 * x - 1).latent_dist.mode()


# ── GT image loader ────────────────────────────────────────────────────────────

def load_gt_image(cam, target_size=None):
    """Load original (unedited) frame from cam.image_path."""
    img = Image.open(cam.image_path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return T.ToTensor()(img).cuda()


# ── Resize helper ──────────────────────────────────────────────────────────────

def resize_for_vae(tensor, resize):
    """Resize (N,3,H,W) to nearest multiple of 64, bounded by resize on long side."""
    N, C, H, W = tensor.shape
    factor = resize / max(W, H)
    nw = int((W * factor) // 64) * 64
    nh = int((H * factor) // 64) * 64
    return F.interpolate(tensor, size=(nh, nw), mode='bilinear', align_corners=False)


# ── Main refinement ────────────────────────────────────────────────────────────

def refine_sds(dataset, pipe, ply_path, prompt,
               guidance_scale=10.5, image_guidance_scale=1.2,
               iterations=800, resize=512, save_path=None,
               sequence_length=2):

    device = torch.device("cuda:0")
    torch_dtype = torch.float16

    # ── Load model ────────────────────────────────────────────────────────────
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
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    gaussians.spatial_lr_scale = scene.cameras_extent

    # ── Freeze dynamic pool (CGaussianModel is not nn.Module, access directly) ─
    for attr in ['_xyz_motion', '_rotation_motion', '_features_dc_motion',
                 '_features_rest_motion', '_opacity_motion',
                 '_opacity_motion_center', '_opacity_motion_var', '_xyz_disp']:
        if hasattr(gaussians, attr):
            getattr(gaussians, attr).requires_grad_(False)

    # ── Static-only optimizer ─────────────────────────────────────────────────
    position_lr = 1.6e-4 * scene.cameras_extent
    l = [
        {'params': [gaussians._xyz],          'lr': position_lr, 'name': 'xyz'},
        {'params': [gaussians._features_dc],   'lr': 2.5e-3,      'name': 'f_dc'},
        {'params': [gaussians._features_rest], 'lr': 2.5e-3/20,   'name': 'f_rest'},
        {'params': [gaussians._opacity],       'lr': 0.05,        'name': 'opacity'},
        {'params': [gaussians._scaling],       'lr': 5e-3,        'name': 'scaling'},
        {'params': [gaussians._rotation],      'lr': 1e-4,        'name': 'rotation'},
    ]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    # ── Get all training cameras ──────────────────────────────────────────────
    train_cams, _ = scene.getTrainCameras(shuffle=False, get_img=False)
    print(f"Training cameras: {len(train_cams)}")

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # ── Load IP2P ─────────────────────────────────────────────────────────────
    print("Loading IP2P pipeline...")
    DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
    IP2P_SOURCE = "timbrooks/instruct-pix2pix"
    tokenizer    = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
    unet         = UNet3DConditionModel.from_pretrained_2d(IP2P_SOURCE, subfolder="unet")

    for m in (vae, text_encoder, unet):
        m.requires_grad_(False)

    vae.to(device, dtype=torch_dtype).enable_slicing()
    text_encoder.to(device, dtype=torch_dtype)
    unet.to(device, dtype=torch_dtype)
    unet.set_attn_processor(AttnProcessor2_0())
    unet.enable_gradient_checkpointing()

    ip2p = InstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler"),
    )

    ip2p.scheduler.config.num_train_timesteps = 1000
    ip2p.scheduler.set_timesteps(20)

    with torch.no_grad():
        prompt_embeds = ip2p._encode_prompt(
            prompt, device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        ).detach()   # [3, 77, 768]

    ip2p.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    print("IP2P ready.")

    # ── SDS loop ──────────────────────────────────────────────────────────────
    progress = tqdm(range(1, iterations + 1), desc="SDS refinement")
    for iteration in progress:
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick sequence_length random cameras
        viewpoint_cams = random.sample(train_cams, sequence_length)

        images, gt_images = [], []
        for cam in viewpoint_cams:
            pkg = render(cam, gaussians, pipe, bg,
                         near=dataset.near, far=dataset.far, mode=0)
            images.append(pkg["render"].unsqueeze(0))
            gt = load_gt_image(cam, target_size=(cam.image_width, cam.image_height))
            gt_images.append(gt.unsqueeze(0))

        image_tensor = torch.cat(images, 0)          # (seq, 3, H, W)
        gt_tensor    = torch.cat(gt_images, 0)       # (seq, 3, H, W)

        # Resize to VAE-friendly resolution
        vae_renders = resize_for_vae(image_tensor.to(torch_dtype), resize)
        vae_gt      = resize_for_vae(gt_tensor.to(torch_dtype), resize)

        torch.cuda.empty_cache()
        latents      = encode_1(ip2p, vae_renders)           # (seq, 4, h/8, w/8)
        with torch.no_grad():
            image_latents = encode_2(ip2p, vae_gt)           # (seq, 4, h/8, w/8)

        latents       = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length).to(torch_dtype)
        image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length).to(torch_dtype)
        uncond_image_latents = torch.zeros_like(image_latents)

        noise = torch.randn_like(latents)
        t = torch.randint(int(1000*0.02), int(1000*0.98), [1], dtype=torch.long, device=device)
        latents_noisy = ip2p.scheduler.add_noise(latents, noise, t)

        with torch.no_grad():
            noise_pred_text  = ip2p.unet(torch.cat([latents_noisy, image_latents], dim=1),
                                         t, prompt_embeds[0:1], None, None, False)[0]
            noise_pred_image = ip2p.unet(torch.cat([latents_noisy, image_latents], dim=1),
                                         t, prompt_embeds[1:2], None, None, False)[0]
            noise_pred_uncond = ip2p.unet(torch.cat([latents_noisy, uncond_image_latents], dim=1),
                                          t, prompt_embeds[2:3], None, None, False)[0]
            noise_pred = (noise_pred_uncond
                          + guidance_scale * (noise_pred_text - noise_pred_image)
                          + image_guidance_scale * (noise_pred_image - noise_pred_uncond))

        alphas = ip2p.scheduler.alphas_cumprod.to(device)
        w = (1 - alphas[t]).view(-1, 1, 1, 1)
        grad = torch.nan_to_num(w * (noise_pred - noise))
        target = (latents - grad).detach().to(torch_dtype)
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / sequence_length

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if iteration % 50 == 0:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path is None:
        prompt_word = re.sub(r'[^\w]', '', prompt.strip().split()[-1]).lower()
        save_path = os.path.join(
            os.path.dirname(ply_path), "..",
            f"point_cloud_refine/{prompt}/point_cloud.ply"
        )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    gaussians.save_ply(save_path)
    print(f"\nSaved refined model to: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    opt   = OptimizationParams(parser)
    pipe  = PipelineParams(parser)
    parser.add_argument("--ply_path",             type=str, required=True)
    parser.add_argument("--prompt",               type=str, required=True)
    parser.add_argument("--guidance_scale",       type=float, default=10.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.2)
    parser.add_argument("--sds_iters",            type=int, default=800)
    parser.add_argument("--resize",               type=int, default=512)
    parser.add_argument("--save_path",            type=str, default=None)
    parser.add_argument("--quiet",                action="store_true")
    args = get_combined_args(parser)
    safe_state(silent=True)

    m = model.extract(args)
    refine_sds(
        m, pipe.extract(args),
        args.ply_path, args.prompt,
        args.guidance_scale, args.image_guidance_scale,
        args.sds_iters, args.resize,
        getattr(args, "save_path", None),
    )
