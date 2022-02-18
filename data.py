import download
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch
import os, cv2
from PIL import Image, ImageOps
from scipy import io
import random

# seed_everything(42, workers=True)

def _get_file_list(data_path):
    """This function detects all image files within the specified parent
       directory for either training or testing. The path content cannot
       be empty, otherwise an error occurs.
    Args:
        data_path (str): Points to the directory where training or testing
                         data instances are stored.
    Returns:
        list, str: A sorted list that holds the paths to all file instances.
    """

    data_list = []

    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".mat")):
                    data_list.append(os.path.join(subdir, file))

    data_list.sort()

    if not data_list:
        raise FileNotFoundError("No data was found")

    return data_list

def _check_consistency(zipped_file_lists, n_total_files):
    """A consistency check that makes sure all files could successfully be
       found and stimuli names correspond to the ones of ground truth maps.
    Args:
        zipped_file_lists (tuple, str): A tuple of train and valid path names.
        n_total_files (int): The total number of files expected in the list.
    """

    assert len(list(zipped_file_lists)) == n_total_files, "Files are missing"

    for file_tuple in zipped_file_lists:
        file_names = [os.path.basename(entry) for entry in list(file_tuple)]
        file_names = [os.path.splitext(entry)[0] for entry in file_names]
        file_names = [entry.replace("_fixMap", "") for entry in file_names]
        file_names = [entry.replace("_fixPts", "") for entry in file_names]

        assert len(set(file_names)) == 1, "File name mismatch"

def _get_random_indices(list_length):
    """A helper function to generate an array of randomly shuffled indices
       to divide the MIT1003 and CAT2000 datasets into training and validation
       instances.
    Args:
        list_length (int): The number of indices that is randomly shuffled.
    Returns:
        array, int: A 1D array that contains the shuffled data indices.
    """

    indices = np.arange(list_length)
    prng = np.random.RandomState(42)
    prng.shuffle(indices)

    return indices

class SaliconDataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "salicon/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")
            download.download_salicon(self.parent_path)

        path = "train/"
        if not self.train:
            path = "val/" 

        self.img_dir  = self.data_path + "stimuli/" + path
        self.gt_dir = self.data_path + "saliency/" + path
        self.fix_dir = self.data_path + "fixations/" + path

        self.img_ids = [nm.split(".")[0] for nm in os.listdir(self.img_dir)]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [640,480]
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            #transforms.RandomAutocontrast(p=0.5),
            #transforms.RandomEqualize(p=0.5),
            #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ColorJitter(
					brightness=0.4,
					contrast=0.4,
					saturation=0.4,
					hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ])
        self.img_transform_val = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])    
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + ".mat")
        
        img = Image.open(img_path).convert('RGB')

        gt = Image.open(gt_path).convert('L')

        fixations = self.mat_loader(fix_path, (self.gt_size[0],self.gt_size[1]))
        fixations = self.pts2pil(fixations, img)

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations= np.array(fixations).astype('float')

        if self.train: 
            img = self.img_transform(img)
        else:
            img = self.img_transform_val(img)

        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)
    
    def __len__(self):
        return len(self.img_ids)    

    def pts2pil(self, fixpts, img):
        fixmap = Image.new("L", img.size)
        for p in fixpts:
            fixmap.putpixel((p[0], p[1]), 255)
        return fixmap

    def mat_loader(self, path, shape):
        mat = io.loadmat(path)["gaze"]
        fix = []
        for row in mat:
            data = row[0].tolist()[2]
            for p in data:
                if p[0]<shape[0] and p[1]<shape[1]: # remove noise at the boundary.
                    fix.append(p.tolist())
        return fix

class TestLoader(DataLoader):
    def __init__(self, img_dir, img_ids):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])    
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert('RGB')
        sz = img.size
        img = self.img_transform(img)
        return img, img_id, sz
    
    def __len__(self):
        return len(self.img_ids)

class Mit1003Dataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "mit1003/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")
            download.download_mit1003(self.parent_path)

        self.img_dir  = self.data_path + "stimuli/" 
        self.gt_dir = self.data_path + "saliency/" 
        self.fix_dir = self.data_path + "fixations/"
        
        self.n_train = 803
        self.n_valid = 200

        list_x = _get_file_list(self.img_dir)
        list_y = _get_file_list(self.gt_dir)
        list_f = _get_file_list(self.fix_dir)

        _check_consistency(zip(list_x, list_y, list_f), 1003)

        indices = _get_random_indices(1003)

        if self.train:
            excerpt = indices[:self.n_train]
        else:
            excerpt = indices[self.n_train:]

        self.lists_x = [list_x[idx] for idx in excerpt]
        self.lists_y = [list_y[idx] for idx in excerpt]
        self.lists_f = [list_f[idx] for idx in excerpt]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384,384]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            # transforms.ColorJitter(
			# 		brightness=0.4,
			# 		contrast=0.4,
			# 		saturation=0.4,
			# 		hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ]) 
        
    def __getitem__(self, idx):
        img_path = self.lists_x[idx]
        gt_path = self.lists_y[idx]
        fix_path = self.lists_f[idx]

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        fixations = Image.open(fix_path).convert('L')

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0],self.gt_size[1]))
        
        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.lists_x)    

class CAT2000Dataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "cat2000/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")
            download.download_cat2000(self.parent_path)

        self.img_dir  = self.data_path + "stimuli/" 
        self.gt_dir = self.data_path + "saliency/" 
        self.fix_dir = self.data_path + "fixations/"
        
        self.n_train = 1600
        self.n_valid = 400

        list_x = _get_file_list(self.img_dir)
        list_y = _get_file_list(self.gt_dir)
        list_f = _get_file_list(self.fix_dir)

        _check_consistency(zip(list_x, list_y, list_f), 2000)

        indices = _get_random_indices(100)

        if self.train:
            # sample uniformly from all 20 categories
            ratio = self.n_train * 100 // 2000
            excerpt = np.tile(indices[:ratio], 20)

            for idx, _ in enumerate(excerpt):
                excerpt[idx] = excerpt[idx] + idx // ratio * 100

        else:
            # sample uniformly from all 20 categories
            ratio = self.n_valid * 100 // 2000
            excerpt = np.tile(indices[-ratio:], 20)

            for idx, _ in enumerate(excerpt):
                excerpt[idx] = excerpt[idx] + idx // ratio * 100

        self.lists_x = [list_x[idx] for idx in excerpt]
        self.lists_y = [list_y[idx] for idx in excerpt]
        self.lists_f = [list_f[idx] for idx in excerpt]

        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384,384]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            # transforms.ColorJitter(
			# 		brightness=0.4,
			# 		contrast=0.4,
			# 		saturation=0.4,
			# 		hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ]) 
        
    def __getitem__(self, idx):
        img_path = self.lists_x[idx]
        gt_path = self.lists_y[idx]
        fix_path = self.lists_f[idx]
        
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')

        fixations = self.mat_loader(fix_path, (self.gt_size[0],self.gt_size[1]))
        #fixations = self.pts2pil(fixations, img)

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = np.flip(fixations)
                # fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0],self.gt_size[1]))
        
        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.lists_x)    

    def pts2pil(self, fixpts, img):
        fixmap = Image.new("L", img.size)
        for p in fixpts:
            fixmap.putpixel((p[0], p[1]), 255)
        return fixmap

    def mat_loader(self, path, shape):
        mat = io.loadmat(path)["fixLocs"]
        return mat

class PASCALSDataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "pascals/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")
            download.download_pascals(self.parent_path)

        self.img_dir  = self.data_path + "stimuli/" 
        self.gt_dir = self.data_path + "saliency/" 
        self.fix_dir = self.data_path + "fixations/"
        
        self.n_train = 650
        self.n_valid = 200

        list_x = _get_file_list(self.img_dir)
        list_y = _get_file_list(self.gt_dir)
        list_f = _get_file_list(self.fix_dir)

        _check_consistency(zip(list_x, list_y, list_f), 850)

        indices = _get_random_indices(850)

        if self.train:
            excerpt = indices[:self.n_train]
        else:
            excerpt = indices[self.n_train:]

        self.lists_x = [list_x[idx] for idx in excerpt]
        self.lists_y = [list_y[idx] for idx in excerpt]
        self.lists_f = [list_f[idx] for idx in excerpt]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384,384]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            # transforms.ColorJitter(
			# 		brightness=0.4,
			# 		contrast=0.4,
			# 		saturation=0.4,
			# 		hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ]) 
        
    def __getitem__(self, idx):
        img_path = self.lists_x[idx]
        gt_path = self.lists_y[idx]
        fix_path = self.lists_f[idx]

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        fixations = Image.open(fix_path).convert('L')

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0],self.gt_size[1]))
        
        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.lists_x)    
    
class OSIEDataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "osie/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")
            download.download_osie(self.parent_path)

        self.img_dir  = self.data_path + "stimuli/" 
        self.gt_dir = self.data_path + "saliency/" 
        self.fix_dir = self.data_path + "fixations/"
        
        self.n_train = 500
        self.n_valid = 200

        list_x = _get_file_list(self.img_dir)
        list_y = _get_file_list(self.gt_dir)
        list_f = _get_file_list(self.fix_dir)

        _check_consistency(zip(list_x, list_y, list_f), 700)

        indices = _get_random_indices(700)

        if self.train:
            excerpt = indices[:self.n_train]
        else:
            excerpt = indices[self.n_train:]

        self.lists_x = [list_x[idx] for idx in excerpt]
        self.lists_y = [list_y[idx] for idx in excerpt]
        self.lists_f = [list_f[idx] for idx in excerpt]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384,384]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            # transforms.ColorJitter(
			# 		brightness=0.4,
			# 		contrast=0.4,
			# 		saturation=0.4,
			# 		hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ]) 
        
    def __getitem__(self, idx):
        img_path = self.lists_x[idx]
        gt_path = self.lists_y[idx]
        fix_path = self.lists_f[idx]

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        fixations = Image.open(fix_path).convert('L')

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0],self.gt_size[1]))
        
        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.lists_x)    

class DUTOMRONDataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "dutomron/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")
            download.download_dutomron(self.parent_path)

        self.img_dir  = self.data_path + "stimuli/" 
        self.gt_dir = self.data_path + "saliency/" 
        self.fix_dir = self.data_path + "fixations/"
        
        self.n_train = 4168
        self.n_valid = 1000

        list_x = _get_file_list(self.img_dir)
        list_y = _get_file_list(self.gt_dir)
        list_f = _get_file_list(self.fix_dir)

        _check_consistency(zip(list_x, list_y, list_f), 5168)

        indices = _get_random_indices(5168)

        if self.train:
            excerpt = indices[:self.n_train]
        else:
            excerpt = indices[self.n_train:]

        self.lists_x = [list_x[idx] for idx in excerpt]
        self.lists_y = [list_y[idx] for idx in excerpt]
        self.lists_f = [list_f[idx] for idx in excerpt]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384,384]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            # transforms.ColorJitter(
			# 		brightness=0.4,
			# 		contrast=0.4,
			# 		saturation=0.4,
			# 		hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ]) 
        
    def __getitem__(self, idx):
        img_path = self.lists_x[idx]
        gt_path = self.lists_y[idx]
        fix_path = self.lists_f[idx]

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        fixations = Image.open(fix_path).convert('L')

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0],self.gt_size[1]))
        
        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.lists_x)    

class FIWIDataset(DataLoader):
    def __init__(self, data_path, train=False, exten='.png', input_size_h=256, input_size_w=256):
        self.data_path = data_path + "fiwi/"
        self.train = train
        if not os.path.exists(self.data_path):
            self.parent_path = os.path.dirname(self.data_path[:-1])
            self.parent_path = os.path.join(self.parent_path, "")
            download.download_fiwi(self.parent_path)

        self.img_dir  = self.data_path + "stimuli/" 
        self.gt_dir = self.data_path + "saliency/" 
        self.fix_dir = self.data_path + "fixations/"
        
        self.n_train = 99
        self.n_valid = 50

        list_x = _get_file_list(self.img_dir)
        list_y = _get_file_list(self.gt_dir)
        list_f = _get_file_list(self.fix_dir)

        _check_consistency(zip(list_x, list_y, list_f), 149)

        indices = _get_random_indices(149)

        if self.train:
            excerpt = indices[:self.n_train]
        else:
            excerpt = indices[self.n_train:]

        self.lists_x = [list_x[idx] for idx in excerpt]
        self.lists_y = [list_y[idx] for idx in excerpt]
        self.lists_f = [list_f[idx] for idx in excerpt]

        self.exten = exten
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        self.gt_size = [384,384]
        self.img_transform = transforms.Compose([
            transforms.Resize((self.input_size_h, self.input_size_w)),
            # transforms.ColorJitter(
			# 		brightness=0.4,
			# 		contrast=0.4,
			# 		saturation=0.4,
			# 		hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ]) 
        
    def __getitem__(self, idx):
        img_path = self.lists_x[idx]
        gt_path = self.lists_y[idx]
        fix_path = self.lists_f[idx]

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        fixations = Image.open(fix_path).convert('L')

        if self.train:
            if random.random() > 0.5:
                img = ImageOps.mirror(img)
                gt = ImageOps.mirror(gt)
                fixations = ImageOps.mirror(fixations)

        gt = np.array(gt).astype('float')
        gt = cv2.resize(gt, (self.gt_size[0],self.gt_size[1]))

        fixations = np.array(fixations).astype('float')
        fixations = cv2.resize(fixations, (self.gt_size[0],self.gt_size[1]))
        
        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')
        
        assert np.min(gt)>=0.0 and np.max(gt)<=1.0
        assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.lists_x)    