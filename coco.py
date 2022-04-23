import os
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, size):
        self.root = root
        # self.transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        # ])
        self.inp_size = size
        self.transforms = transforms.Compose([
                               transforms.Resize(self.inp_size), 
                               transforms.RandomCrop(self.inp_size), 
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10), # Include for resnet
                               transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]
                               )
            ]) 
        self.disp_transforms = transforms.Compose([
                                transforms.Resize(self.inp_size),
                                transforms.RandomCrop(self.inp_size),
                                transforms.ToTensor()
        ])
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.category_map = {
            1: 'person',
            2: 'bicycle',
            3: 'car',
            4: 'motorcycle',
            5: 'airplane',
            6: 'bus',
            7: 'train',
            8: 'truck',
            9: 'boat',
            10: 'traffic light',
            11: 'fire hydrant',
            12: 'stop sign',
            13: 'stop sign',
            14: 'parking meter',
            15: 'bench',
            16: 'bird',
            17: 'cat',
            18: 'dog',
            19: 'horse',
            20: 'sheep',
            21: 'cow',
            22: 'elephant',
            23: 'bear',
            24: 'zebra',
            25: 'giraffe',
            27: 'backpack',
            28: 'umbrella',
            31: 'handbag',
            32: 'tie',
            33: 'suitcase',
            34: 'frisbee',
            35: 'skis',
            36: 'snowboard',
            37: 'sports ball',
            38: 'kite',
            39: 'baseball bat',
            40: 'baseball glove',
            41: 'skateboard',
            42: 'surfboard',
            43: 'tennis racket',
            44: 'bottle',
            46: 'wine glass',
            47: 'cup',
            48: 'fork',
            49: 'knife',
            50: 'spoon',
            51: 'bowl',
            52: 'banana',
            53: 'apple',
            54: 'sandwich',
            55: 'orange',
            56: 'broccoli',
            57: 'carrot',
            58: 'hot dog',
            59: 'pizza',
            60: 'donut',
            61: 'cake',
            62: 'chair',
            63: 'couch',
            64: 'potted plant',
            65: 'bed',
            67: 'dining table',
            70: 'toilet',
            72: 'tv',
            73: 'laptop',
            74: 'mouse',
            75: 'remote',
            76: 'keyboard',
            77: 'cell phone',
            78: 'microwave',
            79: 'oven',
            80: 'toaster',
            81: 'sink',
            82: 'refrigerator',
            84: 'book',
            85: 'clock',
            86: 'vase',
            87: 'scissors',
            88: 'teddy bear',
            89: 'hair drier',
            90: 'toothbrush'
        }


    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        img = img.convert('RGB')

        img_disp = Image.open(os.path.join(self.root, path))
        img_disp = img_disp.convert('RGB')

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        category_ids = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            category_ids.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(category_ids)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Choose box to black out
        if len(category_ids) != 0 :
            box_id = int(torch.rand(1) * len(category_ids))
            boxes = boxes[box_id]
            labels = labels[box_id] - 1 # Shift labels to 0-index
            
            #Make labels one-hot
            one_hot = torch.zeros(90)
            one_hot[labels] = 1.
            labels = one_hot

            # White out region of image corresponding to box
            img = self.white_out_image(img, boxes)
            img_disp = self.white_out_image(img, boxes)
        else : 
            one_hot = torch.zeros(90)
            labels = one_hot

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        # my_annotation["image_id"] = img_id
        # my_annotation["area"] = areas
        # my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation['labels'], self.disp_transforms(img_disp)

    def __len__(self):
        return len(self.ids)

    def white_out_image(self, img, box) :
        
        img = torchvision.transforms.ToTensor()(img)
        img[:, int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 0
        img = torchvision.transforms.ToPILImage()(img)

        return img

    def class_to_label(self, ids) :

        labels_list = []

        for id in ids : 
            labels_list.append(self.category_map[id+1])
        
        return labels_list