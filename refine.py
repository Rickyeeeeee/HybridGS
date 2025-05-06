import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from utils.general_utils import build_scaling_rotation
from gaussian_renderer import render_3dgs, render_bbsplat, network_gui
import sys, time
from scene import Scene, GaussianModel
from scene.refine_model import RefineGaussianModel
from scene.bbsplat_model import BBSplatGaussianModel
from scene.dataset_readers import fetchPly
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, RefinementParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def refinement(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, load_iteration, load_pc_count=100000):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # backup main code
    cmd = f'cp ./refine.py {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=None)
    bbsplat_gaussians = BBSplatGaussianModel(sh_degree=3)
    source_model_ply_path = os.path.join(dataset.source_model_path, dataset.source_ply_name)
    pcd = fetchPly(source_model_ply_path)
    bbsplat_gaussians.create_from_pcd(pcd, scene.cameras_extent, add_sky_box=False)
    bbsplat_gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    bbsplat_gaussians.use_app = opt.exposure_compensation

    if bbsplat_gaussians.use_app:
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_texture_for_log = 0.0
    initial_texture_alpha = bbsplat_gaussians.get_texture_alpha[0:1].detach().clone()
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        xyz_lr = bbsplat_gaussians.update_learning_rate(iteration)

        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            print("Viewpoint: ", viewpoint_cam.image_name)

        gt_image, _ = viewpoint_cam.get_image()
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render_bbsplat(viewpoint_cam, bbsplat_gaussians, pipe, bg)
        
        image: torch.Tensor
        viewspace_point_tensor: torch.Tensor
        visibility_filter: torch.Tensor  # Possibly a BoolTensor (dtype=torch.bool)
        radii: torch.Tensor
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        impact = render_pkg["impact"]
        # image, viewspace_point_tensor, visibility_filter, radii = \
        #     render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        image_ycbcr = image
        gt_image_ycbcr = gt_image

        Ll1 = l1_loss(image_ycbcr, gt_image_ycbcr)
        ssim_map = ssim(image.unsqueeze(0), gt_image.unsqueeze(0), size_average=False)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_map.mean())
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        weights = opt.max_impact_threshold - torch.clamp(impact[visibility_filter], 0, opt.max_impact_threshold)
        textures_reg = (bbsplat_gaussians.get_texture_color[visibility_filter].mean(dim=[1, 2, 3]) * weights).mean() * opt.lambda_texture_value
        textures_reg += torch.abs((bbsplat_gaussians.get_texture_alpha[visibility_filter] - initial_texture_alpha).mean(dim=[1, 2]) * weights).mean() * opt.lambda_alpha_value

        # loss
        total_loss = loss + dist_loss + normal_loss + textures_reg
        # For MCMC sampler
        total_loss += opt.opacity_reg * bbsplat_gaussians.get_texture_alpha.mean()
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
             # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_texture_for_log = 0.4 * textures_reg.item() + 0.6 * ema_texture_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "texture": f"{ema_texture_for_log:.{5}f}",
                    "Points": f"{len(bbsplat_gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_bbsplat, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if opt.texture_from_iter <= iteration < opt.texture_to_iter:
                bbsplat_gaussians.activate_texture_training()

            if iteration >= opt.texture_to_iter:
                bbsplat_gaussians.deactivate_texture_training()

            if iteration > opt.position_lr_max_steps:
                bbsplat_gaussians.deactivate_gaussians_training()

            # Densification
            # if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #     size = len(gaussians.get_texture_alpha)
            #     dead_mask = (gaussians.get_texture_alpha.view(size, -1).mean(1) <= 0.005).squeeze(-1)
            #     gaussians.relocate_gs(dead_mask=dead_mask)
            #     gaussians.add_new_gs(cap_max=opt.cap_max)

            bbsplat_gaussians.set_base_grad_to_zero()

            # Optimizer step
            if iteration < opt.iterations:
                bbsplat_gaussians.optimizer.step()
                bbsplat_gaussians.optimizer.zero_grad(set_to_none = True)

                # L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                # actual_covariance = L @ L.transpose(1, 2)

                # def op_sigmoid(x, k=100, x0=0.995):
                #     return 1 / (1 + torch.exp(-k * (x - x0)))

                # #size = len(gaussians.get_texture_alpha)
                # #opacity = gaussians.get_texture_alpha.view(size, -1).mean(1, keepdim=True) * 10 # Rescale to get maximum = 1
                # opacity = torch.ones([gaussians.get_texture_alpha.shape[0], 1], dtype=torch.float32, device="cuda") # Fix opacity to 1 (results in the paper obtained this way)
                # noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1 - opacity)) * opt.noise_lr * xyz_lr
                # noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                # gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((bbsplat_gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    app_model.save_weights(scene.model_path, opt.iterations+load_iteration)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    rp = RefinementParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--load_iteration', type=int, default=False)
    parser.add_argument('--load_pc_count', type=int, default=100000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1_000, 2_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1_000, 2_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    setup_seed(22)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    refinement(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.load_iteration, 
        args.load_pc_count)

    # All done
    print("\nTraining complete.")