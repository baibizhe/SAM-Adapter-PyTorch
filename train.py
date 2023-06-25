import argparse
import os
import warnings
from collections import OrderedDict

import monai.data
import torchmetrics
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast as autocast, GradScaler

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import  numpy as np
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_MSC_nii_data_loader (spec, tag=''):
    if spec is None:
        return None
    data_list= []
    image_dir_list = os.listdir(spec['args']['root_path_1'])
    label_dir_list = os.listdir(spec['args']['root_path_2'])
    image_dir_list= list(map(lambda x:os.path.join(spec['args']['root_path_1'],x),image_dir_list))
    label_dir_list= list(map(lambda x:os.path.join(spec['args']['root_path_2'],x),label_dir_list))
    print(label_dir_list[0:3],'label_dir_list[0:3]')
    print(image_dir_list[0:3],'image_dir_list[0:3]')

    assert len(image_dir_list) == len(label_dir_list)

    for i in range(len(image_dir_list)):
        data_list.append({'img':image_dir_list[i],'seg':label_dir_list[i]})
    dataset = monai.data.Dataset(data=data_list)

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            if 'shape' in k:
                continue
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_data_num = config['val_datasets']['val_datasets_num']
    val_data_names = config['val_datasets']['val_datasets_name']
    print('val_data_num',val_data_num,'val_data_names',val_data_names)
    val_loaders = []
    for i in range(1,val_data_num+1):
        val_loaders.append(make_data_loader(config['val_datasets'][f'val_dataset{i}'], tag='val'))
    # val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    # datasets_name = ['Kvasir','CVC-ClinicDB','CVC-ColonDB','ETIS-pDB','CVC-300',]
    # if config.get('val_dataset1'):
    #     val_loader1 = make_data_loader(config.get('val_dataset1'), tag='val')
    #     val_loader2 = make_data_loader(config.get('val_dataset2'), tag='val')
    #     val_loader3 = make_data_loader(config.get('val_dataset3'), tag='val')
    #     val_loader4 = make_data_loader(config.get('val_dataset4'), tag='val')
    #     return  train_loader, [val_loader,val_loader1,val_loader2,val_loader3,val_loader4] ,datasets_name
    return train_loader, val_loaders,val_data_names

#
# def eval_psnr(loader, model, eval_type=None):
#     model.eval()
#
#     if eval_type == 'f1':
#         metric_fn = utils.calc_f1
#         metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
#     elif eval_type == 'fmeasure':
#         metric_fn = utils.calc_fmeasure
#         metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
#     elif eval_type == 'ber':
#         metric_fn = utils.calc_ber
#         metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
#     elif eval_type == 'cod':
#         metric_fn = utils.calc_cod
#         metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
#
#     if local_rank == 0:
#         pbar = tqdm(total=len(loader), leave=False, desc='val')
#     else:
#         pbar = None
#
#     pred_list = []
#     gt_list = []
#     for batch in loader:
#         for k, v in batch.items():
#             batch[k] = v.cuda()
#
#         inp = batch['inp']
#
#         pred = torch.sigmoid(model.infer(inp))
#
#         batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
#         batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]
#
#         dist.all_gather(batch_pred, pred)
#         pred_list.extend(batch_pred)
#         dist.all_gather(batch_gt, batch['gt'])
#         gt_list.extend(batch_gt)
#         if pbar is not None:
#             pbar.update(1)
#
#     if pbar is not None:
#         pbar.close()
#
#     pred_list = torch.cat(pred_list, 1)
#     gt_list = torch.cat(gt_list, 1)
#     result1, result2, result3, result4 = metric_fn(pred_list, gt_list)
#
#     return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()

    return 2 * volume_intersect / volume_sum

def eval_segment(loader, model, args,writer=None,epcoh=0):
    model.eval()

    metric1 = 'dice'
    metric2 ='iou'
    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    dice_result=0
    total_num_samples = 0
    jaccard_fun=torchmetrics.JaccardIndex(task='binary', num_classes=2)
    jaccard_score = 0
    desired_w = args.val_img_w
    desired_h = args.val_img_h
    with torch.no_grad():
        for idx,batch in enumerate(loader):
            inp_original_cpu = batch['inp']
            gt_original_cpu = batch['gt']
            for k, v in batch.items():
                if 'shape' in k :
                    continue
                batch[k] = v.cuda()
            inp=batch['inp']

            with torch.autocast(device_type = 'cuda'):
                pred = torch.sigmoid(model.infer(inp,gt_original_cpu))
            total_num_samples += pred.shape[0]
            # print(pred.min(),pred.max(),gt_original_cpu.min(),gt_original_cpu.max())

            if pbar is not None:
                pbar.update(1)
            if (desired_w != 1024) or (desired_h!=1024):
                pred_desired = torch.nn.functional.interpolate(pred.cpu().float(),(desired_w,desired_h))
                target_desized = torch.nn.functional.interpolate(gt_original_cpu.float(),(desired_w,desired_h))
            else:
                pred_desired = pred.cpu().float()
                target_desized = gt_original_cpu.float()

            pred_desired = torch.where(pred_desired < 0.5, 0, 1)
            target_desized = torch.where(target_desized < 0.5, 0, 1)
            dice_result+=compute_dice_coefficient(pred_desired[:,0,:,].unsqueeze(1)>0,target_desized>0 )
            jaccard_score+=jaccard_fun(pred_desired[:,0,:,].unsqueeze(1).cpu()>0,target_desized.cpu()>0)
            # print(pred_desired.shape,target_desized.shape,gt_original_cpu.shape,pred.shape)

            if idx < 10:
                writer.add_images(tag=f"images/{idx}/valid_input_images", img_tensor=inp.cpu().numpy(),
                                  global_step=epcoh)
                writer.add_images(tag=f"images/{idx}/valid_predicte_1024_result", img_tensor=(pred>0.5).cpu().detach().numpy(),
                                  global_step=epcoh)
                writer.add_images(tag=f"images/{idx}/valid_predicted_desired_size", img_tensor=pred_desired.cpu().detach().numpy(),
                                  global_step=epcoh)
                writer.add_images(tag=f"images/{idx}/valid_label_images", img_tensor=batch['gt'].cpu().numpy(),
                                  global_step=epcoh)

    if pbar is not None:
        pbar.close()

    dice_result  = dice_result/total_num_samples
    jaccard_score = jaccard_score/total_num_samples
    return  dice_result,jaccard_score,np.array(0),np.array(0),metric1,metric2,'none','none'
def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start

def prepare_training_adapter():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start


def train(train_loader, model):
    model.train()
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            if isinstance(v,torch.Tensor):
                batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        # print(inp.shape,gt.shape)

        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)
        # break
    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)
def exec_fun(model,input):
    model(**input)

def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loaders,val_datasets_name = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    epoch_max, eval_per_epoch, start_eval_e = get_custom_epoch(len(train_loader.dataset))

    model, optimizer, epoch_start = prepare_training()
    model.optimizer = optimizer
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer=optimizer,T_mult=2,T_0=3)
    lr_scheduler = CosineAnnealingLR(optimizer, epoch_max, eta_min=config.get('lr_min'))

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])

    model.load_state_dict(sam_checkpoint, strict=False)


    # current_model_dict = model.state_dict()
    # new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
    #                   zip(current_model_dict.keys(), sam_checkpoint.values())}
    # new_state_dict = OrderedDict(new_state_dict)
    #
    # message = model.load_state_dict(new_state_dict, strict=False)
    # print(message)
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    for name, para in model.named_parameters():
        if "obj_model" in name :
            para.requires_grad_(False)
    # for name, para in model.named_parameters():
    #     if para.requires_grad ==   True:
    #         print(name)


    # else:
        #     print(name)

            # ============================================================
            # ============================================================

    # #
    # current_model_dict = model.state_dict()
    # warnings.warn('loading state dict is not strcit')
    # new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
    #                   zip(current_model_dict.keys(), sam_checkpoint.values())}
    # new_state_dict = OrderedDict(new_state_dict)
    # pos_embed = sam_checkpoint['image_encoder.pos_embed']
    # pos_embed = torch.nn.functional.interpolate(pos_embed.permute(0, 3, 1, 2), scale_factor=(1 / 2, 1 / 2),
    #                                             mode='bicubic').permute(0, 2, 3, 1)
    # sam_checkpoint['image_encoder.pos_embed'] = pos_embed
    # message = model.load_state_dict(new_state_dict, strict=False)
    # print(message)
    #
    # #
    # # message = sam.load_state_dict(new_state_dict, strict=False)
    # for name, para in model.named_parameters():
    #     if ("image_encoder" in name and "prompt_generator"  in name) or ("image_encoder" in name and "rel_pos"  in name)   \
    #          or ("image_encoder" in name and "Adapter"  in name) or  ("mask_decoder" in name ):
    #         para.requires_grad_(True)
    #         print(name)
    #
    #     else:
    #         para.requires_grad_(False)



    #
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    # model =  torch.compile(model,mode='max-autotune')

    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    early_stop_e = 0
    early_stop_e_max = 50
    best_dice =0
    assert len(val_loaders) == len(val_datasets_name),'assert len(val_loaders) == len(val_datasets_name)'

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        # train_loss_G = 0

        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            log_info.append('lr={:.5f}'.format(optimizer.param_groups[0]['lr']))

            writer.add_scalar('train_loss',  train_loss_G, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')
        epoch_m_dice = 0
        epoch_m_iou = 0
        total_samples = 0
        if (epoch_val is not None) and (epoch >= start_eval_e )  and (epoch%eval_per_epoch ==0) :
        # if (epoch_val is not None) and (epoch >= 0 )  and (epoch%1 ==0) :
            torch.cuda.empty_cache()
            for val_loader_idx in range(len(val_loaders)):
                dice_cur, iou_cur, result3, result4, metric1, metric2, metric3, metric4 = eval_segment(val_loaders[val_loader_idx], model,args=args,writer=writer,epcoh=epoch,
                    )
                d_name = val_datasets_name[val_loader_idx]
                epoch_m_dice+=dice_cur*len(val_loaders[val_loader_idx].dataset)
                epoch_m_iou+=iou_cur*len(val_loaders[val_loader_idx].dataset)
                total_samples+=len(val_loaders[val_loader_idx].dataset)
                if local_rank == 0:
                    log_info.append('val {}: {}={:.4f}'.format(d_name,metric1, dice_cur))
                    writer.add_scalar(f'val_{d_name}_'+metric1,  dice_cur, epoch)
                    log_info.append(' {}={:.4f}'.format(metric2, iou_cur))
                    writer.add_scalar(f'val_{d_name}_' + metric2, iou_cur, epoch)
                    log_info.append('\n')


            epoch_m_iou /= total_samples
            epoch_m_dice /= total_samples
            log_info.append('val {}: ={:.4f}'.format('m_iou', epoch_m_iou))
            log_info.append('\n')
            log_info.append('val {}: ={:.4f}'.format('m_dice', epoch_m_dice))
            log_info.append('\n')

            writer.add_scalar(f'val_epoch_m_iou_' , epoch_m_iou, epoch)
            writer.add_scalar(f'val_epoch_m_dice_', epoch_m_dice, epoch)
            if epoch_m_dice > best_dice:
                        best_dice = epoch_m_dice
                        early_stop_e = 0
            else:
                early_stop_e+=1
                if early_stop_e >early_stop_e_max:
                    break


        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all,optimizer.param_groups[0]['lr']))

        log(', '.join(log_info))
        writer.flush()


def get_custom_epoch(len_of_tranin_data):
    if len_of_tranin_data < 200:
        start_eval_e = 20
        eval_per_epoch = 3
        epoch_max = 100
    if 200<=len_of_tranin_data <= 1000:
        start_eval_e = 15
        eval_per_epoch = 2
        epoch_max = 60
    if 1000<=len_of_tranin_data:
        start_eval_e = 5
        eval_per_epoch = 1
        epoch_max = 40
    return epoch_max, eval_per_epoch, start_eval_e


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--lr', type=float,default=0.001)

    parser.add_argument('--freqnums', type=float,default=0.25)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--train_img_dir", type=str, help="")
    parser.add_argument("--train_label_dir", type=str,  help="")
    parser.add_argument("--val_img_dir", type=str, help="")
    parser.add_argument("--val_label_dir", type=str,  help="")
    parser.add_argument("--val_img_w", type=int)
    parser.add_argument("--val_img_h", type=int)
    parser.add_argument("--epoch_max", type=int, default=30, help="")

    torch.cuda.manual_seed_all(2)
    torch.manual_seed(2)
    args = parser.parse_args()
    if not args.val_img_w:
        args.val_img_w= 1024
    if not args.val_img_h:
        args.val_img_h= 1024
        warnings.warn('does not provide desired valid image size , using 1024x 1024')
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')
    config['optimizer']['args']['lr'] = args.lr
    config['lr_min'] = args.lr/10
    # config['model']['args']['encoder_mode']['freq_nums'] = args.freqnums
    if args.train_img_dir:
        config['train_dataset']['dataset']['args']['root_path_1'] = args.train_img_dir
        config['train_dataset']['dataset']['args']['root_path_2'] = args.train_label_dir
    if args.val_img_dir:
        config['val_datasets']['val_dataset1']['dataset']['args']['root_path_1'] = args.val_img_dir
        config['val_datasets']['val_dataset1']['dataset']['args']['root_path_2'] = args.val_label_dir
    config['epoch_max'] = args.epoch_max
    print(args)

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
