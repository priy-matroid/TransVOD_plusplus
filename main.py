# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# TransVOD++
# Copyright (c) 2022 Shanghai Jiao Tong University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) SenseTime. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import datasets

import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import wandb
from pycocotools.coco import COCO
 

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--pretrainedmodel')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--img_side', default=600, type=int)
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--freeze_spatial', default=False, action='store_true')


    
    parser.add_argument('--num_ref_frames', default=3, type=int, help='number of reference frames')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--checkpoint', default=False, action='store_true')


    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int)
    parser.add_argument('--interval1', default=20, type=int)
    parser.add_argument('--interval2', default=60, type=int)

    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--data_root')
    parser.add_argument('--dataset_file', default='vid_multi')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--vid_path', default='./data/vid', type=str)
    parser.add_argument('--coco_pretrain', default=False, action='store_true')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    wandb.init(project="UAV_higher_res_single", name='_'.join(args.output_dir.split('/')))
    if args.dataset_file == "vid_single":
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
    else:
        from engine_multi import evaluate, train_one_epoch
        import util.misc_multi as utils
        # from engine_multi_mm import evaluate, train_one_epoch
        # import util.misc_mm as utils
    print(args.dataset_file)
    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    for n, k in model.named_parameters():
        print(n, k.shape)

    # dataset_train = build_dataset(image_set='train_joint', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)

    dataset_train = build_dataset(image_set='train_vid', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    

    if args.distributed:
        print("11111")
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(f"{n}{p.shape}")
        

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    print(args.lr_drop_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop_epochs)

    # optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
    #                                   weight_decay=args.weight_decay)
    
    # from torch.optim.lr_scheduler import LambdaLR
    # import math


    

    # def lr_lambda(current_step, total_steps=len(data_loader_train)*args.epochs, warmup_steps=len(data_loader_train)*int(0.2*args.epochs)):
    #     if current_step < warmup_steps:
    #         return float(current_step) / float(max(1, warmup_steps))
    #     else:
    #         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    #         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)




    if args.distributed:
        print('**************************DISTRIBUTED***************')
        print(f'device ids {args.gpu}')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    
    filtered_ids = []
    base_ds_img_ids = base_ds.getImgIds()
    base_ds_img_infos = base_ds.loadImgs(base_ds_img_ids)
    for info in base_ds_img_infos:
        if info['is_vid_train_frame']:
            filtered_ids.append(info['id'])
    
    filtered_ann_data = {
            'categories': base_ds.dataset['categories'],
            'images': [img for img in base_ds.dataset['images'] if img['id'] in filtered_ids],
            'annotations': [ann for ann in base_ds.dataset['annotations'] if ann['image_id'] in filtered_ids]
                }

    filtered_json_path = './base_ds_filtered.json'
    with open(filtered_json_path, 'w') as f:
        json.dump(filtered_ann_data, f)
    
    base_ds = COCO(filtered_json_path)
    os.remove(filtered_json_path)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    # if args.pretrainedmodel:
    #     # checkpoint = torch.load(args.pretrainedmodel)
    #     # tmp_dict = checkpoint['model']
    #     # for name, param in model_without_ddp.named_parameters(): # TODO PRIY
    #     #     if ('temp' in name):
    #     #         param.requires_grad = True
    #     #     elif ('dynamic' in name):
    #     #         param.requires_grad = True
    #     #     else:
    #     #         param.requires_grad = False
    #     checkpoint = torch.load(args.pretrainedmodel)
    #     checkpoint_state_dict = checkpoint['model']
    #     curr_model_state_dict = model_without_ddp.state_dict()
    #     conflict_params = []
    #     for param_name in checkpoint_state_dict.keys():
    #         if param_name in curr_model_state_dict.keys():
    #             # need to check if same size
    #             loaded_tensor_shape = checkpoint_state_dict[param_name].shape
    #             current_model_tensor_shape = curr_model_state_dict[param_name].shape
    #             if loaded_tensor_shape != current_model_tensor_shape:
    #                 print(f"loaded checkpoint {param_name} is size {loaded_tensor_shape} and current model is size {current_model_tensor_shape}")
    #                 conflict_params.append(param_name)
    #         else:
    #             print(f"loaded: {param_name} not in current model")
    #     for conflict_param in conflict_params:
    #         checkpoint_state_dict.pop(conflict_param)



    #     missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint_state_dict, strict=False)
    #     print("********************PRETRAINED LOADED*******************")
    #     unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    #     if len(missing_keys) > 0:
    #         print('Missing Keys: {}'.format(missing_keys))
    #     if len(unexpected_keys) > 0:
    #         print('Unexpected Keys: {}'.format(unexpected_keys))


    #     for name, param in model_without_ddp.named_parameters():
    #         if args.freeze_spatial:
    #             if ('temp' in name):
    #                 param.requires_grad = True
    #             elif ('dynamic' in name):
    #                 param.requires_grad = True
    #             else:
    #                 param.requires_grad = False
    #         elif args.freeze_backbone:
    #             if 'backbone' in name:
    #                 param.requires_grad = False
    #         else:
    #             continue
                
    #     total_params_model = 0
    #     total_params_with_grad = 0
    #     for name, param in model_without_ddp.named_parameters():
    #         if param.requires_grad:
    #             total_params_with_grad += 1
    #         total_params_model += 1
    #     print("***************************************************************")
    #     print(f"TOTAL PARAMS: {total_params_model}")
    #     print(f"PARAMS WITH GRAD: {total_params_with_grad}")
    #     print("***************************************************************")

    

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if args.eval:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        else:
            tmp_dict = model_without_ddp.state_dict().copy()
            if args.coco_pretrain: # single frame baseline
                for k, v in checkpoint['model'].items():
                    if ('class_embed' not in k):
                        tmp_dict[k] = v 
                    else:
                        print('k', k)
            else:
                # multi-frame (TransVOD++)
                tmp_dict = checkpoint['model']
                for name, param in model_without_ddp.named_parameters():
                    if ('temp' in name):
                        param.requires_grad = True
                    elif ('dynamic' in name):
                        param.requires_grad = True
                    elif ('bbox_embed' in name):
                        param.requires_grad = True
                    elif ('class_embed' in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(tmp_dict, strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        
        total_params_model = 0
        total_params_with_grad = 0
        for name, param in model_without_ddp.named_parameters():
            if param.requires_grad:
                total_params_with_grad += 1
            total_params_model += 1
        print("***************************************************************")
        print(f"TOTAL PARAMS: {total_params_model}")
        print(f"PARAMS WITH GRAD: {total_params_with_grad}")
        print("***************************************************************")

    if args.eval: # TODO
        test_stats, coco_evaluator, overall_result = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args.data_root)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        
        output_path = os.path.join(args.output_dir, f"final_predicitions_{'_'.join(args.output_dir.split('/'))}.json")
        print('saving results')
        with open(output_path, 'w') as json_file:
            json.dump(overall_result, json_file)
        print('done saving')
        return

    # if args.eval or True: # TODO
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir, args.data_root)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        print('args.output_dir', args.output_dir)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        #test_stats, coco_evaluator = evaluate(
         #   model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        #)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch%1==0:

            test_stats, coco_evaluator, overall_result = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args.data_root)
        

            # cocoDt = coco_evaluator.coco_eval['bbox'].cocoDt
            # cocoGt = coco_evaluator.coco_eval['bbox'].cocoGt

            # cocoDt_dict = cocoDt.dataset
            output_path = os.path.join(args.output_dir, f"cocoDt_epoch{epoch}.json")
            print('saving results')
            with open(output_path, 'w') as json_file:
                json.dump(overall_result, json_file)
            print('done saving')

            wandb.log({"val_class_error": test_stats['class_error']})
            wandb.log({"val_loss": test_stats['loss']})
            wandb.log({"val_loss_ce": test_stats['loss_ce']})
            wandb.log({"val_loss_bbox": test_stats['loss_bbox']})
            wandb.log({"val_loss_giou": test_stats['loss_giou']})
            wandb.log({"map50": test_stats['coco_eval_bbox'][1]})


        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
