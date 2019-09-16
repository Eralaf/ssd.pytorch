"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot

================================
Based on the file `voc0712.py`

Developped by Florian Napierala
================================
"""

import torch
import sys
import cv2
import os

import torch.utils.data as data
import os.path          as osp
import numpy            as np

from .config            import HOME

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree  as ET


# /!\ TODO put these parameters in a config file rather than a python script...
CLASSES = (  # always index 0
          'numero','pantographe') # TODO change the classes.

# note: if you used our download scripts, this should be right
P_ROOT = osp.join(HOME, "data/cafeine/")


class PersonnalVOCAnnotationTransform(object):
    """
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        - class_to_ind      (dict, optional): dictionary lookup of classnames -> indexes
        - keep_difficult    (bool, optional): keep difficult instances or not
    """

    def __init__(self,
                 class_to_ind   = None,
                 keep_difficult = False):

        self.class_to_ind   = class_to_ind or dict( zip(CLASSES, range(len(CLASSES))) )
        self.keep_difficult = keep_difficult

    def __call__(self,
                 target,
                 width,
                 height):
        """
        Arguments:
            - target        (annotation) : the target annotation to be made usable will be an ET.Element
            - height        (int):         height
            - width         (int):         width
        Returns:
            A list containing lists of bounding boxes  [bbox coords, class name]
        """

        res = []

        for obj in target.iter('object'):

            difficult = int(obj.find('difficult').text) == 1

            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts    = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []

            for i, pt in enumerate(pts):

                cur_pt = int(bbox.find(pt).text) - 1

                # scale height or width => cur_pt is between [0,1]
                if i % 2 == 0 : # xmin or xmax
                    cur_pt = float(cur_pt) / width
                else :          # ymin or ymax
                    cur_pt = float(cur_pt) / height

                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]

            bndbox.append(label_idx)

            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class PersonnalDetection(data.Dataset):
    """
    Personnal Detection Dataset Object

    input is image, target is annotation

    Arguments:

        - root              (string):             filepath to VOCdevkit folder.
        - extension         (string):             default : "JPG"
        - transform         (callable, optional): transformation to perform on the input image
        - target_transform  (callable, optional): transformation to perform on the target `annotation` (eg: take in caption string, return tensor of word indices)
        - dataset_name      (string, optional):   which dataset to load (in config.py) (default: 'siara')
    """

    def __init__(self,
                 root,
                 extension        = "JPG", # TODO should take an array instead if several extensions...
                 transform        = None,
                 target_transform = PersonnalVOCAnnotationTransform(),
                 dataset_name     = 'siara'):

        self.root             = root
        self.transform        = transform
        self.target_transform = target_transform
        self.name             = dataset_name
        self._annopath        = osp.join('%s', 'Annotations', '%s.{}'.format("xml"))     # TODO add the needed architecture on the readme
        self._imgpath         = osp.join('%s', 'Images',      '%s.{}'.format(extension)) # TODO add the needed architecture on the readme
        self.ids              = list()

        for name in os.listdir(osp.join(root,'Annotations')):
            # as we have only one annotations file per image
            self.ids.append((root,name.split('.')[0]))


    """
        Get image, ground truth from an index
    """
    def __getitem__(self, index):

        im, gt, h, w = self.pull_item(index)
        return im, gt

    """
        Get dataset size
    """
    def __len__(self):

        return len(self.ids)

    """
        # TODO comment this function
    """
    def pull_item(self, index):

        img_id = self.ids[index]
        target = ET.parse(  self._annopath % img_id).getroot()
        img    = cv2.imread(self._imgpath  % img_id)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:

            target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]     # BGR (as cv2 open an image) => RGB
            # img = img.transpose(2, 0, 1) # equivalent - seems slower but prettier

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width # TODO identify why a permutation is required
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
