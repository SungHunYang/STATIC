import torch
import cv2
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import os, sys, time
import argparse
import numpy as np
from tqdm.auto import tqdm
import random
# from networks_pixelformer.PixelFormer import PixelFormer
from model_S.get_model import *
# from model.get_model_bin import *
from einops import rearrange
from sobel import Sobel
import warnings

# UserWarning을 무시, 지금 backward 문제가 있긴 한 듯
warnings.filterwarnings("ignore", category=UserWarning)
from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, block_print, enable_print, \
    normalize_result, inv_normalize, convert_arg_line_to_args
from loss import *

# torch.backends.cudnn.benchmark = True
# Remove explicit setting of CUDA_VISIBLE_DEVICES, let PyTorch manage devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4" # Remove or set via command line if necessary # 0,1,2,3,4

parser = argparse.ArgumentParser(description='PixelFormer PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--model_name', type=str, help='model name', default='pixelformer')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
parser.add_argument('--pretrain', type=str, help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--gt_path', type=str, help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq', type=int, help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                    action='store_true')
parser.add_argument('--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate', type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus', type=float,
                    help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
                    default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate', help='if set, will perform random rotation for augmentation',
                    action='store_true')
parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                    action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size', type=int, help='number of nodes for distributed training', default=1)
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)
parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='env://')
parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')
parser.add_argument('--gpu', type=str, help='GPU ids to use, e.g., "0,1,2"', default='0,1,2,3,4')
parser.add_argument('--multiprocessing_distributed', help='Use multi-processing distributed training to launch '
                                                          'N processes per node, which has N GPUs. This is the '
                                                          'fastest way to use PyTorch for either single node or '
                                                          'multi node data parallel training', action='store_true',default=True )
# Online eval
parser.add_argument('--do_online_eval', help='if set, perform online eval in every eval_freq steps',
                    action='store_true')
parser.add_argument('--data_path_eval', type=str, help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval', type=str, help='path to the groundtruth data for online evaluation',
                    required=False)
parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file for online evaluation',
                    required=False)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq', type=int, help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory', type=str, help='output directory for eval summary,'
                                                               'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0

    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if name not in list(own_state.keys()):
            ckpt_name.append(name)
        else:
            try:
                own_state[name].copy_(param)
                cnt += 1
            except:
                continue

    print('#reused param : {} / {}\n'.format(cnt, len(state_dict.items())))

    return model


def online_eval(global_step, model, dataloader_eval, gpu, ngpus, post_process=False):
    eval_measures = torch.zeros(10).cuda(device=gpu)

    random_save_idx = random.randint(0, len(dataloader_eval.data) - 2)

    for idx, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            next_image = torch.autograd.Variable(eval_sample_batched['next_image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            next_depth = eval_sample_batched['next_depth']

            image = torch.cat((image.unsqueeze(1), next_image.unsqueeze(1)), dim=1).contiguous()
            gt_depth = torch.cat((gt_depth.unsqueeze(1), next_depth.unsqueeze(1)), dim=1).contiguous()
            t = image.shape[1]

            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            pred_depth, norm_est, A, norm_mask, = model(image)
            pred_depth = rearrange(pred_depth, 'b t c h w -> (b t) c h w')

            if post_process:
                image = rearrange(image, 'b t c h w -> (b t) c h w')
                image_flipped = flip_lr(image)
                image_flipped = rearrange(image_flipped, '(b t) c h w -> b t c h w', t=t)
                pred_depth_flipped, norm_est, A, norm_mask = model(image_flipped)
                pred_depth_flipped = rearrange(pred_depth_flipped, 'b t c h w -> (b t) c h w')
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = rearrange(pred_depth, '(b t) c h w -> b t c h w', t=t)

            if idx == random_save_idx and gpu == 0:
                save_path = args.log_directory + '/' + args.model_name + '/online_eval' + f'/{global_step}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                norm_mask = norm_mask[0, 0, :, :].cpu().numpy()
                norm_mask = (norm_mask * 255.0).astype(np.uint8)
                A = A[0,0,:, :].cpu().numpy()
                A = (A * 255.0).astype(np.uint8)

                cv2.imwrite(save_path + '/mask_before_{}.png'.format(global_step), norm_mask)
                cv2.imwrite(save_path + '/mask_gt_{}.png'.format(global_step), A)

                for _t in range(t):
                    input_image_i = image_flipped[0, _t, :, :, :]
                    input_image_i = inv_normalize(input_image_i)
                    input_image_i = np.transpose(input_image_i.cpu().numpy(), (1, 2, 0))
                    input_image_i = (input_image_i * 255.0).astype(np.uint8)
                    input_image_i = cv2.cvtColor(input_image_i, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path + '/input_{}_{}.png'.format(global_step, _t), input_image_i)

                    pred_norm_i = norm_est[0, _t, :, :, :].cpu().numpy()
                    pred_norm_i = np.transpose(pred_norm_i, (1, 2, 0))
                    pred_norm_i = (pred_norm_i + 1.0) / 2.0 * 255.0
                    pred_norm_i = pred_norm_i.astype(np.uint8)
                    cv2.imwrite(save_path + '/norm_{}_{}.png'.format(global_step, _t), pred_norm_i)

                    pred_depth_i = pred_depth[0, _t, :, :, :].cpu().numpy()
                    pred_depth_i = np.clip(pred_depth_i, args.min_depth_eval, args.max_depth_eval)
                    pred_depth_i = (pred_depth_i - args.min_depth_eval) / (args.max_depth_eval - args.min_depth_eval)
                    pred_depth_i = ((1.0 - pred_depth_i) * 255.0).astype(np.uint8)
                    pred_depth_i = np.transpose(pred_depth_i, (1, 2, 0))
                    pred_depth_i = pred_depth_i[:, :, 0]
                    pred_depth_i = cv2.applyColorMap(pred_depth_i, cv2.COLORMAP_INFERNO)
                    cv2.imwrite(save_path + '/depth_{}_{}.png'.format(global_step, _t), pred_depth_i)

            pred_depth = pred_depth[:, 0, :, :, :]
            gt_depth = gt_depth[:, 0, :, :, :]

            pred_depth = pred_depth.cpu().numpy().squeeze().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))

        with open(f'/home/mvpcoin/lee/depthmodel/logs/log.txt', 'a+') as f:
            f.write('TEST Global STEP : {}'.format(global_step))
            f.write("\n======================================================================================\n")
            config_1 = 'Computing errors for {} eval samples'.format(int(cnt))
            f.write(config_1 + '\n')
            config_2 = "{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel',
                                                                                              'log10',
                                                                                              'rms',
                                                                                              'sq_rel', 'log_rms',
                                                                                              'd1',
                                                                                              'd2',
                                                                                              'd3')
            f.write(config_2 + '\n')
            for i in range(8):
                form = '{:7.5f}, '.format(eval_measures_cpu[i])
                f.write(form)
            form = '{:7.5f}\n'.format(eval_measures_cpu[8])
            # f.write(form + '\n')
            f.write(form)
            f.write("======================================================================================\n\n")

        return eval_measures_cpu

    return None


def make_sobel(depth_sobel):
    B, c, h, w = depth_sobel.shape
    sobel_class = Sobel(3)
    sobel = sobel_class.compute_normals_from_depth(depth_sobel)
    sobel = F.interpolate(sobel, (int(h // 4), int(w // 4)), mode='bilinear', align_corners=True)
    return sobel


def main_worker(gpu, ngpus_per_node, args):
    if args.gpu is None:
        args.gpu = gpu
    else:
        if isinstance(args.gpu, list):
            args.gpu = args.gpu[gpu % len(args.gpu)]
        else:
            args.gpu = int(args.gpu)

    # Initialize process group for distributed training
    if args.distributed:
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)

    # Build and compile model
    model = build_model()
    # Optionally use torch.compile for JIT compilation (PyTorch 2.0 feature)
    # model = torch.compile(model)

    # check_point = torch.load('/home/mvpcoin/lee/depthmodel/logs/pixelformer_kittieigen/networks/model-6948-best_abs_rel_0.06372.pth', map_location='cpu')['model']
    # model = load_my_state_dict(model, check_point)


    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        encoder_num_params = sum([np.prod(p.size()) for p in model.encoder.parameters()])
        print("== Encoder number of parameters: {:,}".format(encoder_num_params))
        decoder_num_params = sum([np.prod(p.size()) for p in model.decoder.parameters()])
        print("== Decoder number of parameters: {:,}".format(decoder_num_params))
        video = sum([np.prod(p.size()) for p in model.video.parameters()])
        print("== Video number of parameters: {:,}".format(video))
        norm = sum([np.prod(p.size()) for p in model.norm_encoder.parameters()])
        print("== norm number of parameters: {:,}".format(norm))
        # norm = sum([np.prod(p.size()) for p in model.normal.parameters()])
        # print("== norm number of parameters: {:,}".format(norm))
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        # print("== Else number of parameters: {:,}".format(num_params - (decoder_num_params + encoder_num_params + norm + video)))
        print("== Total number of parameters: {:,}\n".format(num_params))

    model.train()

    # Load checkpoint if provided
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    # Move model to GPU
    model = model.cuda(args.gpu)

    # Use DistributedDataParallel for all cases
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        if len(args.gpu) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(args.gpu)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        [{'params': [param for name, param in model.named_parameters() if 'encoder' in name], 'lr': args.learning_rate},
         {'params': [param for name, param in model.named_parameters() if 'encoder' not in name],
          'lr': args.learning_rate * 2.}], lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)

    # Dataloader
    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    # Loss functions
    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    huber_loss = nn.SmoothL1Loss()

    num_epochs = args.num_epochs
    global_step = 0
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.08 * args.learning_rate
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    start = global_step // steps_per_epoch

    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)
    for epoch in range(start, num_epochs):
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader.data, desc=f"Epoch {epoch}")
        for step, sample_batched in enumerate(progress_bar):
            optimizer.zero_grad()

            image = sample_batched['image'].cuda(args.gpu, non_blocking=True)
            next_image = sample_batched['next_image'].cuda(args.gpu, non_blocking=True)
            depth_gt = sample_batched['depth'].cuda(args.gpu, non_blocking=True)
            next_depth_gt = sample_batched['next_depth'].cuda(args.gpu, non_blocking=True)
            fill_gt = sample_batched['fill'].cuda(args.gpu, non_blocking=True)
            next_fill_gt = sample_batched['next_fill'].cuda(args.gpu, non_blocking=True)

            image = torch.cat([image.unsqueeze(1), next_image.unsqueeze(1)], dim=1).contiguous()
            depth_gt = torch.cat([depth_gt.unsqueeze(1), next_depth_gt.unsqueeze(1)], dim=1).contiguous()
            fill_gt = torch.cat([fill_gt.unsqueeze(1), next_fill_gt.unsqueeze(1)], dim=1).contiguous()

            b, t, c, h, w = depth_gt.shape
            fill_gt = rearrange(fill_gt, 'b t c h w -> (b t) c h w')

            norm_ori = make_sobel(fill_gt)
            norm_ori = rearrange(norm_ori, '(b t) c h w -> b t c h w', t=t)

            depth_est, norm_est, normal_mask, gen_mask = model(image)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0

            depth_loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            sobel_loss = huber_loss(norm_est, norm_ori) * 10.0
            norm_mask = torch.cat((normal_mask,1-normal_mask),dim=1)
            mask_loss = huber_loss(norm_mask, gen_mask) * 10.0

            total_loss = depth_loss + sobel_loss + mask_loss

            total_loss.backward()

            current_lr = (args.learning_rate - end_learning_rate) * (
                        1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
            optimizer.param_groups[0]['lr'] = current_lr
            current_lr = (args.learning_rate * 2. - end_learning_rate) * (
                        1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
            optimizer.param_groups[1]['lr'] = current_lr

            optimizer.step()
            current_lr_0 = optimizer.param_groups[0]['lr']
            current_lr_1 = optimizer.param_groups[1]['lr']

            progress_bar.set_description(
                f"Epoch {epoch} [ lr: {current_lr_0:.6f} / {current_lr_1:.6f} ], Loss: {depth_loss.item():.4f}, NORM : {sobel_loss.item():.4f}, MASK : {mask_loss.item():.4f}")

            global_step += 1

            # Online evaluation
            if args.do_online_eval and global_step % args.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(global_step, model, dataloader_eval, args.gpu, ngpus_per_node,
                                                post_process=True)

                if eval_measures is not None:
                    for i in range(9):
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}.pth'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + args.model_name + '/networks' + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}.pth'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'model': model.state_dict()}
                            # torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/networks' + model_save_name)
                            torch.save(checkpoint, args.log_directory + args.model_name + '/networks' + model_save_name)
                model.train()


def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    # Ensure the output directories exist
    os.makedirs(os.path.join(args.log_directory, args.model_name), exist_ok=True)

    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')
        command = 'cp pixelformer/train.py ' + aux_out_path
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp pixelformer/networks/*.py ' + networks_savepath
        os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && cp pixelformer/dataloaders/*.py ' + dataloaders_savepath
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.gpu is not None:
        args.gpu = [int(id) for id in args.gpu.split(',')]
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
        ngpus_per_node = len(args.gpu)
    else:
        args.gpu = [0]  # 기본적으로 첫 번째 GPU를 사용합니다.
        ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(0, ngpus_per_node, args)


if __name__ == '__main__':
    main()
