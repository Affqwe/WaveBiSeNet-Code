from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from losses.fusion_losses import IoULoss,WeightedDiceLoss
import torch.distributed as dist
# from models.DY_SAMPLE_BISE import BiSeNet
from models_compare.esnet import ESNet
from models_compare.canet import CANet
from models_compare.aglnet import AGLNet
from models_compare.bisenetv2 import BiSeNetv2
# from models.LENet import LETNet
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
from numpy import *
from data_loader.dataset_jpg import train_dataset, colorize_mask, fast_hist,val_dataset,compute_iou,calculate_f1_score, calculate_kappa
# from models.ATTENTION_UNET import AttU_Net
# from models.ATTENTION_UNET import AttU_Net
# from models.egeunet import EGEUNet

from models_compare.icnet import ICNet
from models.edanet import EDANet
from models_compare.fssnet import FSSNet
from models.mobile_unet import UNet
from models.res_unet import MultiResUnet
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

import torch.optim as optim
class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr, factor=0.75, loss_patience=10, miou_patience=8, cooldown=10, min_lr=0.000000001, forced_epochs=300):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.factor = factor
        self.loss_patience = loss_patience
        self.miou_patience = miou_patience
        self.best_val_loss = float('inf')
        self.best_miou = 0
        self.loss_no_improve_epochs = 0
        self.miou_no_improve_epochs = 0
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.lr_history = []  # 记录学习率变化
        self.min_lr = min_lr  # 设置最小学习率
        self.forced_epochs = forced_epochs  # 强制运行的最小 epoch 数

    def step(self, val_loss, miou, epoch):
        # 冷却期
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # 平滑指标
        self.best_val_loss = min(self.best_val_loss, val_loss)
        self.best_miou = max(self.best_miou, miou)

        # 检查 mIoU 条件
        if miou > self.best_miou:
            self.miou_no_improve_epochs = 0
        else:
            self.miou_no_improve_epochs += 1

        # 检查 val_loss 条件
        if val_loss < self.best_val_loss:
            self.loss_no_improve_epochs = 0
        else:
            self.loss_no_improve_epochs += 1

        # 学习率调整
        if (self.miou_no_improve_epochs >= self.miou_patience or self.loss_no_improve_epochs >= self.loss_patience) and epoch >= self.forced_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
            print(f"Learning rate reduced. Epoch: {epoch}, New LR: {self.optimizer.param_groups[0]['lr']:.10f}")
            self.cooldown_counter = self.cooldown  # 重置冷却期

        # 即使达到最小学习率，也强制运行到指定轮数
        if epoch < self.forced_epochs:
            print(f"Epoch {epoch}: Forced to continue training (even at min_lr).")

        # 记录学习率
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
def visualize_and_save_top_feature_maps(feature_map, epoch, save_dir=r'F:\巴彦淖尔市\tezhengtu', num_top_channels=3):
    """
    可视化和保存特征最明显的三个通道。

    参数：
        feature_map (numpy.ndarray): 要可视化的特征图，形状为 (num_samples, num_channels, height, width)。
        epoch (int): 当前的 epoch 数，用于保存文件命名。
        save_dir (str): 保存特征图的目录。
        num_top_channels (int): 要保存的最明显的通道数。
    """
    # 检查输入的四维数组
    num_samples, num_channels, height, width = feature_map.shape

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 对每个样本进行特征图可视化和保存
    for sample_index in range(num_samples):
        selected_feature_map = feature_map[sample_index]  # 选择第 sample_index 个样本的特征图

        # 计算每个通道的激活值和
        channel_sums = np.sum(selected_feature_map, axis=(1, 2))  # 计算每个通道上的激活值和，结果为 (num_channels,)

        # 找出激活值和最大的三个通道索引
        top_channels = np.argsort(channel_sums)[-num_top_channels:]  # 取最大的三个通道索引

        # 可视化并保存每个最明显的通道
        for i, channel_index in enumerate(top_channels):
            top_feature_map = selected_feature_map[channel_index]  # 选择当前通道的特征图

            # 可视化并保存特征图
            plt.imshow(top_feature_map, cmap='viridis')
            plt.axis('off')
            save_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{sample_index}_top_channel_{i}_heatmap.png")
            plt.savefig(save_path)
            plt.close()
def manage_txt_files(out_path, create, targettxt=None, value=None):
    if create:
        # Ensure the directory exists
        os.makedirs(out_path, exist_ok=True)

        # List of file names to create
        filenames = [
            "train_loss.txt", "train_acc.txt", "val_loss.txt",
            "val_acc.txt", "val_miou.txt", "val_kappa.txt", "val_f1score.txt"
        ]

        # Create the files
        for filename in filenames:
            file_path = os.path.join(out_path, filename)
            # Create empty file if it doesn't exist
            if not os.path.exists(file_path):
                with open(file_path, 'w') as file:
                    pass  # Just create an empty file

    else:
        if targettxt is None or value is None:
            raise ValueError("Both targetxt and value must be provided when create is False.")

        # Ensure target file exists
        target_path = os.path.join(out_path, targettxt)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"The file {targettxt} does not exist in {out_path}.")

        # Open the target file, write the value, and close the file
        with open(target_path, 'a') as file:
            file.write(f"{value}\n")
def save_val_loss(val_loss, file_path):
    with open(file_path, 'a') as f:
        f.write(f"{val_loss}\n")
parser = argparse.ArgumentParser(description='Training a UNet model')
parser.add_argument('--batch_size', type=int, default=16, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--val_batch_size', type=int, default=8, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--mini_batch_size',type=int,default=8,help='minabatch')
parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=2, help='equivalent to numclass')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.99, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool,default=True, help='enables cuda. default=True')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=8, help='how many threads of cpu to use while loading data')
parser.add_argument('--size_w', type=int, default=512, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=512, help='scale image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')    #翻转图片
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')
parser.add_argument('--pretrained_model', type=str, default='', help='path to pre-trained model')
parser.add_argument('--data_path', default=r'/root/autodl-tmp/code/seg/data/data1210/train', help='path to training images')
parser.add_argument('--val_data_path', default=r'/root/autodl-tmp/code/seg/data/data1210/val', help='path to validation images')
parser.add_argument('--outf', default='./checkpoint/resunet')
parser.add_argument('--save_epoch', default=1, help='path to save model')
parser.add_argument('--test_step', default=512, help='path to val images')   #测试步数多少此显示一次loss
parser.add_argument('--log_step', default=1024, help='path to val images')
parser.add_argument('--num_GPU', default=1, help='number of GPU')
opt = parser.parse_args()
try:
    os.makedirs(opt.outf)
    os.makedirs(opt.outf + '/model/')
except OSError:
    pass
if opt.manual_seed is None:
    opt.manual_seed = 1435
random.seed(opt.manual_seed)
cudnn.benchmark = True

train_datatset_ = train_dataset(opt.data_path, opt.size_w, opt.size_h, opt.flip)
train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.num_workers, prefetch_factor=2)
val_datatset_ =val_dataset(opt.val_data_path, opt.size_w, opt.size_h, opt.flip)
val_loader = torch.utils.data.DataLoader(dataset=val_datatset_, batch_size=opt.batch_size, shuffle=False,
                                   num_workers=opt.num_workers, prefetch_factor=2)
def weights_init(m):
    # 遍历模块时，检查模块名是否包含 'context_path'
    for name, module in m.named_modules():
        if 'context_path' not in name:  # 如果模块名中不包含 'context_path'，则初始化
            if isinstance(module, nn.Conv2d):  # 如果是卷积层
                nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.BatchNorm2d):  # 如果是 BatchNorm2d 层
                module.eps = 1e-5  # 设置 epsilon 值
                module.momentum = 0.1  # 设置 momentum 值
                nn.init.constant_(module.weight, 1.0)  # 初始化 gamma 权重
                nn.init.constant_(module.bias, 0.0)  # 初始化 beta 偏置
# def weights_init(m):
#     # 遍历模块时，初始化所有模块
#     for name, module in m.named_modules():
#         if isinstance(module, nn.Conv2d):  # 如果是卷积层
#             nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
#             if module.bias is not None:
#                 nn.init.constant_(module.bias.data, 0.0)
#         elif isinstance(module, nn.BatchNorm2d):  # 如果是 BatchNorm2d 层
#             module.eps = 1e-5  # 设置 epsilon 值
#             module.momentum = 0.1  # 设置 momentum 值
#             nn.init.constant_(module.weight, 1.0)  # 初始化 gamma 权重
#             nn.init.constant_(module.bias, 0.0)  # 初始化 beta 偏置
net  = MultiResUnet(3,2)
if opt.net != '':
    state_dict = torch.load(r'/root/autodl-tmp/unet/checkpoint/Unet/model/netG_20.pth')
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # 如果state_dict的键中有'module.'前缀，需要移除
    net.load_state_dict(state_dict)  # 这里要修改
else:
    net.apply(weights_init)
if opt.cuda:
    net.cuda()
if opt.num_GPU > 1:
    net = nn.DataParallel(net)
'''
for name, param in net.named_parameters():
    if 'conv' in name:
        print(f"Layer: {name}, Weight: {param.data.mean()}")
'''
###########   LOSS & OPTIMIZER   ##########
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=25)
criterion2 = WeightedDiceLoss(num_classes=2,weight=[1.0,1.0])
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99),weight_decay=0.001,eps=1e-8)
# scheduler = CustomLRScheduler(
#     optimizer=optimizer,
#     initial_lr=opt.lr,
#     factor=0.8,
#     loss_patience=100,
#     miou_patience=15,
#     cooldown=20,
#     min_lr=1e-8,
#     forced_epochs=300  # 强制运行至少 300 个 epoch
# )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='max',
#     factor=0.85,
#     patience=15,
#     threshold=1e-8,
#     threshold_mode='rel',
#     cooldown=8,
#     min_lr=1e-9,
#     eps=1e-9,
#     verbose=False
# )
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True)
###########   GLOBAL VARIABLES   ###########
initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
initial_image = Variable(initial_image)
semantic_image = Variable(semantic_image)
val_initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
val_semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)

if opt.cuda:
    initial_image = initial_image.cuda()
    semantic_image = semantic_image.cuda()
    val_initial_image = val_initial_image.cuda()
    val_semantic_image = val_semantic_image.cuda()
    criterion = criterion.cuda()
    criterion2 = criterion2.cuda()
if __name__ == '__main__':
    start = time.time()
    hist = np.zeros((opt.output_nc, opt.output_nc))
    val_hist = np.zeros((opt.output_nc, opt.output_nc))
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    val_miou_history = []
    val_kappa_history = []
    val_f1score_history = []
    out_txtpath = r"/root/autodl-tmp/code/seg/checkpoint/resunet/logs"
    manage_txt_files(out_path=out_txtpath, create=True)
    for epoch in range(1, opt.niter + 1):
        train_acc_LIST = []
        train_loss_LIST = []
        net.train()
        loader = iter(train_loader)
        for i in range(0, train_datatset_.__len__(), opt.batch_size):

            initial_image_, semantic_image_, name = next(loader)
            initial_image.resize_(initial_image_.size()).copy_(initial_image_)
            semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
            semantic_image_pred= net(initial_image)

            # initial_image = initial_image.view(-1)
            # semantic_image_pred = semantic_image_pred.view(-1)
            ### loss ###
            # from IPython import embed;embed()
            assert semantic_image_pred.size()[2:] == semantic_image.size()[1:]
            loss1_0 = criterion(semantic_image_pred, semantic_image.long())
            loss2_0 = criterion2(semantic_image_pred, semantic_image.long())
            loss = 0.5*loss1_0+0.5*loss2_0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### evaluate ###
            predictions = semantic_image_pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts = semantic_image.data[:].squeeze_(0).cpu().numpy()
            hist += fast_hist(label_pred=predictions.flatten(), label_true=gts.flatten(),
                              num_classes=opt.output_nc)
            train_acc = np.diag(hist).sum() / hist.sum()
            ########### Logging ##########
            if i % opt.log_step == 0:
                train_loss_LIST.append(loss.item())
                train_acc_LIST.append(train_acc)
            if i % opt.test_step == 0:
                gt = semantic_image[0].cpu().numpy().astype(np.uint8)
                gt_color = colorize_mask(gt)
                predictions = semantic_image_pred.data.max(1)[1].squeeze_(1).cpu().numpy()
                prediction = predictions[0]
                predictions_color = colorize_mask(prediction)
                width, height = opt.size_w, opt.size_h
                save_image = Image.new('RGB', (width * 2, height))
                save_image.paste(gt_color, box=(0 * width, 0 * height))
                save_image.paste(predictions_color, box=(1 * width, 0 * height))
                save_image.save(opt.outf + '/epoch_%03d_%03d_gt_pred.png' % (epoch, i))
        train_loss = np.mean(train_loss_LIST)
        train_acc = np.mean(train_acc_LIST)
        manage_txt_files(out_path=out_txtpath, create=False, targettxt="train_loss.txt", value=str(train_loss))
        manage_txt_files(out_path=out_txtpath, create=False, targettxt="train_acc.txt", value=str(train_acc))
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        print(f'epoch{epoch}')
        print(f'train_loss{train_loss}')
        print(f'train_acc{train_acc}')
        net.eval()
        with torch.no_grad():
            loader = iter(val_loader)
            losses = []
            val_acc_list = []
            val_miou_list = []
            val_f1_list = []
            val_kappa_list = []
            epoch_fps = []
            total_time=0
            total_samples=0
            for i in range(0, val_datatset_.__len__(), opt.batch_size):

                val_initial_image_, val_semantic_image_, name = next(loader)
                val_initial_image.resize_(val_initial_image_.size()).copy_(val_initial_image_)
                val_semantic_image.resize_(val_semantic_image_.size()).copy_(val_semantic_image_)
                torch.cuda.synchronize()
                batch_start_time = time.time()
                # 前向传播
                val_semantic_image_pred = net(val_initial_image)
                # 同步 GPU，结束计时
                torch.cuda.synchronize()
                batch_end_time = time.time()
                # 计算时间
                batch_time = batch_end_time - batch_start_time
                total_time += batch_time
                total_samples += val_initial_image_.size(0)  # 累计样本数
                # initial_image = initial_image.view(-1)
            # semantic_image_pred = semantic_image_pred.view(-1)
            ### loss ###
            # from IPython import embed;embed()
                assert val_semantic_image_pred.size()[2:] == val_semantic_image.size()[1:]
                loss1 = criterion(val_semantic_image_pred, val_semantic_image.long())
                loss3 = criterion2(val_semantic_image_pred, val_semantic_image.long())
                loss = 0.5 * loss1 +  0.5 * loss3
                losses.append(loss.item())
                val_predictions = val_semantic_image_pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
                val_gts = val_semantic_image.data[:].squeeze_(0).cpu().numpy()
                # 进行卡方检验
                val_hist += fast_hist(label_pred=val_predictions.flatten(), label_true=val_gts.flatten(),
                                  num_classes=opt.output_nc)
                val_acc = np.diag(val_hist).sum() / val_hist.sum()
                val_acc_list.append(val_acc)
                f1 = calculate_f1_score(val_hist)
                miou = compute_iou(val_hist)
                kappa = calculate_kappa(val_hist)
                val_miou_list.append(miou)
                val_f1_list.append(f1)
                val_kappa_list.append(kappa)
            val_loss = np.mean(losses)
            val_acc_avg = np.mean(val_acc_list)
            val_miou = np.mean(val_miou_list)
            val_kappa = np.mean(val_kappa_list)
            val_f1 = np.mean(val_f1_list)
            manage_txt_files(out_path=out_txtpath, create=False, targettxt="val_acc.txt", value=str(val_acc_avg))
            manage_txt_files(out_path=out_txtpath, create=False, targettxt="val_loss.txt", value=str(val_loss))
            manage_txt_files(out_path=out_txtpath, create=False, targettxt="val_miou.txt", value=str(val_miou))
            manage_txt_files(out_path=out_txtpath, create=False, targettxt="val_f1score.txt", value=str(val_f1))
            manage_txt_files(out_path=out_txtpath, create=False, targettxt="val_kappa.txt", value=str(val_kappa))
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc_avg)
            val_miou_history.append(val_miou)
            val_kappa_history.append(val_kappa)
            val_f1score_history.append(val_f1)
            FPS = total_samples/total_time
            # scheduler.step(val_loss, miou, epoch)
            scheduler.step(val_miou)
            print(f'val_loss:{val_loss}')
            print(val_acc_avg)
            print(f'val_kappa{val_kappa}')
            print(f'val_miou{val_miou}')
            print(f'val_f1{val_f1}')
            print(f'FPS{FPS}')
        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(), '%s/model/netG_%s.pth' % (opt.outf, str(epoch)))
    output_dir = r'/root/autodl-tmp/code/seg/checkpoint/resunet/plots'  # 替换为你的目标文件夹路径
    os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

    # 假设以下是历史记录
    num_epochs = len(train_loss_history)
    epochs = range(1, num_epochs + 1)  # 横轴：epoch 数

    # 1. 绘制训练损失和验证损失在同一张图上
    plt.figure()
    plt.plot(epochs, train_loss_history, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss_history, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))  # 保存到指定文件夹
    plt.close()

    # 2. 绘制训练准确率和验证准确率在同一张图上
    plt.figure()
    plt.plot(epochs, train_acc_history, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_acc_history, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))  # 保存到指定文件夹
    plt.close()

    # 3. 绘制验证 mIoU
    plt.figure()
    plt.plot(epochs, val_miou_history, label='Validation mIoU', marker='d')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Validation mIoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'miou_curve.png'))  # 保存到指定文件夹
    plt.close()

    # 4. 绘制验证 Kappa 系数
    plt.figure()
    plt.plot(epochs, val_kappa_history, label='Validation Kappa', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa')
    plt.title('Validation Kappa')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'kappa_curve.png'))  # 保存到指定文件夹
    plt.close()

    # 5. 绘制验证 F1-score
    plt.figure()
    plt.plot(epochs, val_f1score_history, label='Validation F1-score', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('Validation F1-score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1score_curve.png'))  # 保存到指定文件夹
    plt.close()
    print(f"所有图片已保存到文件夹: {output_dir}")
    end = time.time()
    torch.save(net.state_dict(), '%s/model/netG_final.pth' % opt.outf)