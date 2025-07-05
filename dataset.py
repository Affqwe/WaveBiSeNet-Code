import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import random
import torchvision.transforms as transforms
from osgeo import gdal,gdalconst

def calculate_kappa(hist):
    """
    通过混淆矩阵计算 Kappa 系数
    hist: 混淆矩阵 (num_classes x num_classes)
    """
    # 总样本数
    total = np.sum(hist)

    # 实际分类准确率 p_o
    po = np.diag(hist).sum() / total

    # 计算每个类的实际分布和预测分布
    actual = np.sum(hist, axis=1)  # 行和，代表实际类别分布
    predicted = np.sum(hist, axis=0)  # 列和，代表预测类别分布

    # 计算随机分类的预期准确率 p_e
    pe = np.sum(actual * predicted) / (total ** 2)

    # 计算 Kappa 系数
    kappa = (po - pe) / (1 - pe)

    return kappa
def calculate_f1_score(hist):
    # True Positives: 每个类别正确分类的数量 (对角线)
    tp = np.diag(hist)

    # Precision: tp / (tp + fp)
    # False Positives: 每列的总和 - True Positives
    fp = hist.sum(axis=0) - tp

    # Recall: tp / (tp + fn)
    # False Negatives: 每行的总和 - True Positives
    fn = hist.sum(axis=1) - tp

    # Precision = tp / (tp + fp)
    precision = tp / (tp + fp + 1e-6)  # 加上 1e-6 防止除零错误

    # Recall = tp / (tp + fn)
    recall = tp / (tp + fn + 1e-6)

    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # Calculate mean F1 score across all classes
    f1_score = np.nanmean(f1)  # Use nanmean to ignore any NaN values
    return f1_score
def compute_iou(hist):
    ious = []

    # 对于每个类别计算 IoU
    for i in range(hist.shape[0]):
        TP = hist[i, i]
        FP = np.sum(hist[:, i]) - TP
        FN = np.sum(hist[i, :]) - TP
        denominator = TP + FP + FN

        # 避免分母为 0 的情况
        if denominator == 0:
            iou = np.nan  # 若该类别没有数据，则将 IoU 设为 NaN
        else:
            iou = TP / denominator
        ious.append(iou)

    # 计算 mIoU，忽略 NaN 值
    miou = np.nanmean(ious)
    return miou

def normalize_array_per_channel(arr):
    # 假设输入为 (height, width, channels)
    normalized_arr = np.zeros_like(arr, dtype=np.float32)

    # 遍历每个通道进行归一化
    for c in range(arr.shape[2]):
        channel = arr[:, :, c]
        min_val = np.min(channel)
        max_val = np.max(channel)

        # 避免除零
        if max_val != min_val:
            normalized_arr[:, :, c] = (channel - min_val) / (max_val - min_val)
        else:
            # 如果max_val和min_val相等，将整个通道设置为0
            normalized_arr[:, :, c] = 0.0

    return normalized_arr
def is_image_file(filename):  # 定义一个判断是否是图片的函数
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".tif","geotiff"])

def trans_to_tensor(pic):  # 定义一个转变图像格式的函数
    if isinstance(pic, np.ndarray):
        pic = normalize_array_per_channel(pic)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))  # transpose和reshape区别巨大
        return img.float()


def data_augment(img1, img2, flip=1, ROTATE_90=1, ROTATE_180=1, ROTATE_270=1, add_noise=1):
    n = flip + ROTATE_90 + ROTATE_180 + ROTATE_270 + add_noise
    a = random.random()
    if flip == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if ROTATE_90 == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    if ROTATE_180 == 1:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    if ROTATE_270 == 1:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    if add_noise == 1:
        pass

## 2020/10/26
import torchvision.transforms as transforms
mean_std = ([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
# 将input变成tensor
input_transform = transforms.Compose([
    transforms.ToTensor(),      ##如果是numpy或者pil image格式，会将[0,255]转为[0,1]，并且(hwc)转为(chw)
    transforms.Normalize(*mean_std)     #[0,1]  ---> 符合imagenet的范围[-2.117,2.248][,][,]
])
# 将label变成tensor
def function_label(x):
    if x == 0:
        return 0
    elif 30 < x < 40:
        return 1
    elif 70 < x < 80:
        return 2
    elif 0.7< x <1.2:
        return 1
    elif 1.7 < x < 2.4:
        return 2
    else:
        return 0
class RGBToGray(object):
    def __call__(self, mask):
        mask = mask.convert("L")
        mask = mask.point(function_label)      #在这里处理了mask掩膜文件
        return mask
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()
target_transform = transforms.Compose([
    RGBToGray(),
    MaskToTensor()
])
palette = [255,255,255,
           0,255,0,
           255,0,0,
           255,255,255,
           ]           #在这里进行调色处理
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
## end 2020/10/26

class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=272, size_h=272, flip=0):
        super(train_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/src/') if is_image_file(x)]
        self.list2 = [x for x in os.listdir(data_path + '/label/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        file_name = os.path.splitext(self.list[index])[0]
        filename = file_name+'.png'
        initial_path = os.path.join(self.data_path + '/src/', self.list[index])
        semantic_path = os.path.join(self.data_path + '/label/',filename)
        assert os.path.exists(semantic_path)
        try:
            initial_image = gdal.Open(initial_path,gdalconst.GA_ReadOnly)        #如果要修改为tiff图片的话就要修改这里
            initial_image = initial_image.ReadAsArray()
            initial_image = np.array(initial_image)
            semantic_image = Image.open(semantic_path)
        except OSError:
            return None, None, None

        initial_image = np.resize(initial_image, (self.size_w, self.size_h, initial_image.shape[0]))
        semantic_image = semantic_image.resize((272,272), Image.BILINEAR)

        if self.flip == 1:
            a = random.random()
            if a < 1 / 3:
                initial_image = np.flip(initial_image, axis=2)
                semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                if a < 2 / 3:
                    initial_image = np.rot90(initial_image, k=1, axes=(1, 2))
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)

        initial_image = trans_to_tensor(initial_image)  # 0到1之间 # -1到1之间
        # semantic_image = trans_to_tensor(semantic_image)
        # semantic_image = semantic_image.mul_(2).add_(-1)

        #initial_image = input_transform(initial_image)
        semantic_image = target_transform(semantic_image)

        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)

class pre_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=272, size_h=272):
        super(pre_dataset, self).__init__()
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
    def data(self):
        initial_image = gdal.Open(self.data_path, gdalconst.GA_ReadOnly)  # 如果要修改为tiff图片的话就要修改这里
        geotransform = initial_image.GetGeoTransform()
        projection = initial_image.GetProjection()
        initial_image = initial_image.ReadAsArray()
        initial_image = np.array(initial_image)
        initial_image = np.resize(initial_image, (self.size_w, self.size_h, initial_image.shape[0]))
        initial_image = trans_to_tensor(initial_image)  # 0到1之间
        return initial_image,geotransform,projection
class val_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=272, size_h=272, flip=0):
        super(val_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/src/') if is_image_file(x)]
        self.list2 = [x for x in os.listdir(data_path + '/label/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        file_name = os.path.splitext(self.list[index])[0]
        filename = file_name + '.png'
        initial_path = os.path.join(self.data_path + '/src/', self.list[index])
        semantic_path = os.path.join(self.data_path + '/label/', filename)
        assert os.path.exists(semantic_path)
        try:
            initial_image = gdal.Open(initial_path,gdalconst.GA_ReadOnly)        #如果要修改为tiff图片的话就要修改这里
            initial_image = initial_image.ReadAsArray().astype(int)
            initial_image = np.array(initial_image)
            semantic_image = Image.open(semantic_path)
        except OSError:
            return None, None, None

        initial_image = np.resize(initial_image, (self.size_w, self.size_h, initial_image.shape[0]))
        semantic_image = semantic_image.resize((272, 272), Image.BILINEAR)

        if self.flip == 1:
            a = random.random()
            if a < 1 / 3:
                initial_image = np.flip(initial_image, axis=2)
                semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                if a < 2 / 3:
                    initial_image = np.rot90(initial_image, k=1, axes=(1, 2))
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)

        initial_image = trans_to_tensor(initial_image)  # 0到1之间
        # semantic_image = trans_to_tensor(semantic_image)
        # semantic_image = semantic_image.mul_(2).add_(-1)

        #initial_image = input_transform(initial_image)
        semantic_image = target_transform(semantic_image)

        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)