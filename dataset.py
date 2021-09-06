import glob
from PIL import Image
import torch
from torchvision import transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, src_root, tgt_root, img_size, mode='train'):
        self.src_root = src_root
        self.tgt_root = tgt_root
        self.image_src_names = [x.split("/")[-1] for x in glob.glob(f"{self.src_root}/*.png")]
        self.image_tgt_names = [x.split("/")[-1] for x in glob.glob(f"{self.tgt_root}/*.png")]
        size = (img_size, img_size)
        if mode == 'train':
            self.img_src_transforms = transforms.Compose([transforms.Resize(size=size),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        else:
            self.img_src_transforms = transforms.Compose([transforms.Resize(size=size),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        self.img_mask_transforms = transforms.Compose([transforms.Resize(size=size), transforms.ToTensor()])

    def __getitem__(self, index):
        src = Image.open(f"{self.src_root}/{self.image_src_names[index]}")
        tgt = Image.open(f"{self.tgt_root}/{self.image_tgt_names[index]}")
        src_img = self.img_src_transforms(src)
        tgt_img = self.img_mask_transforms(tgt)
        return src_img, tgt_img

    def get_file_name(self, index):
        return self.image_src_names[index]

    def __len__(self):
        return len(self.image_src_names)