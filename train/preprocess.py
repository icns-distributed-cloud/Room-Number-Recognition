import csv
import cv2
import h5py
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

class Preprocessor:
    '''
    Preprocessor for SVHN dataset(format 1)
    Idea from https://github.com/ArtemGits/cnn_svhn_full_numbers
    '''
    DEFAULT_TRAIN_FOLDER_PATH = './datasets/train'
    DEFAULT_TEST_FOLDER_PATH = './datasets/test'
    DEFAULT_EXTRA_FOLDER_PATH = './datasets/extra'
    DEFAULT_TRAIN_NEW_FOLDER_PATH = './datasets/train_new'
    DEFAULT_TEST_NEW_FOLDER_PATH = './datasets/test_new'
    DEFAULT_EXTRA_NEW_FOLDER_PATH = './datasets/train_new'
    DEFAULT_CSV_NAME = 'labels.csv'
    
    def __init__(self, file_path, mode='train'):
        '''
        Initializer for FullLoader
        Opens digitStruct.mat file
        '''

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            self.f = h5py.File(file_path, 'r')
            if mode == 'train':
                self.folder = self.DEFAULT_TRAIN_FOLDER_PATH
                self.folder_new = self.DEFAULT_TRAIN_NEW_FOLDER_PATH
            elif mode == 'extra':
                self.folder = self.DEFAULT_EXTRA_FOLDER_PATH
                self.folder_new = self.DEFAULT_EXTRA_NEW_FOLDER_PATH
            else:
                self.folder = self.DEFAULT_TEST_FOLDER_PATH
                self.folder_new = self.DEFAULT_TEST_NEW_FOLDER_PATH
            self.name = self.f['digitStruct']['name']
            self.bbox = self.f['digitStruct']['bbox']
        except Exception as exc:
            raise "Error raised from FullLoader.__init__() : " + str(exc)
        
    def get_name(self, idx):
        '''
        Returns "digitStruct(idx)/name"
        '''
        return ''.join([chr(c[0]) for c in self.f[self.name[idx][0]]])
    
    def get_bbox(self, idx):
        '''
        Returns "digitStruct(idx)/bbox"
        '''
        bbox = dict()
        it = self.bbox[idx].item()
        bbox['height'] = self._attr(self.f[it]['height'])
        bbox['left'] = self._attr(self.f[it]['left'])
        bbox['top'] = self._attr(self.f[it]['top'])
        bbox['width'] = self._attr(self.f[it]['width'])
        bbox['label'] = self._attr(self.f[it]['label'])

        return bbox

    def get_merged_bbox(self, idx):
        '''
        Returns merged bbox
        '''
        topleft = [0, 0]
        btmright = [0, 0]
        bbox = self.get_bbox(idx)
        length = len(bbox['label'])

        topleft[0] = bbox['left'][0]
        topleft[1] = min(bbox['top'])

        btmright[0] = bbox['left'][length - 1] + bbox['width'][length - 1]
        _max_top_val = 0
        _max_top_idx = 0
        for i in range(length):
            if bbox['top'][i] >= _max_top_val:
                _max_top_val = bbox['top'][i]
                _max_top_idx = i
        btmright[1] = _max_top_val + bbox['height'][_max_top_idx]

        return {
            'height': btmright[1] - topleft[1],
            'left': topleft[0],
            'top': topleft[1],
            'width': btmright[0] - topleft[0],
            'label': self._make_label(bbox['label'])
        }
    

    def crop_image(self, idx, bbox):
        filename = self.get_name(idx)
        img = Image.open('{}/{}'.format(self.folder, filename))
        cropped = img.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height']))
        resized = cropped.resize((48, 48))
        resized.save('{}/e{}'.format(self.folder_new, filename))
        # cropped.save('{}/{}'.format(self.folder_new, filename))

    def save_csv(self, idx, bbox):
        f = open('{}/{}'.format(self.folder_new, self.DEFAULT_CSV_NAME), 'a', newline='')
        wr = csv.writer(f)
        row = ['e' + self.get_name(idx)]
        row.extend(bbox['label'])
        wr.writerow(row)
        f.close()
    
    def _attr(self, attr):
        if len(attr) > 1:
            attr = [self.f[attr[j].item()][0][0] for j in range(len(attr))]
        else:
            attr = [attr[0][0]]
        return attr

    def _make_label(self, label_list):
        result = [0, 0, 10, 10, 10, 10]
        if len(label_list) > 0:
            result[0] = 1
            result[1] = min(len(label_list), 4)
            for idx, val in enumerate(label_list):
                try:
                    result[2 + int(idx)] = int(val)
                except IndexError:
                    break
        return result

if __name__ == "__main__":
    pp = Preprocessor('./datasets/extra/digitStruct.mat', mode='extra')
    for idx in tqdm(range(37706, 202353), ncols=80):
        bbox = pp.get_merged_bbox(idx)
        pp.crop_image(idx, bbox)
        pp.save_csv(idx, bbox)