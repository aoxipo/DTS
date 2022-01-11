from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
from utils import *
import cv2

#TYPE_TEST = "DEBUG"
TYPE_TEST = "false"
TYPE_TEST = "DEBUG"
TYPE_TEST = "false"
class DataGenerate():
    def __init__(self, img_shape = 256, data_set_number = "M",image_path = './TMTSJ0312.fits', density_path  = "./TMTSJ0312_density_area_2.h5", label_path = "./TMTSJ0312.cat"):
        self.read_map(image_path)
        self.read_density(density_path)
        self.read_label(label_path)
        self.img_shape = img_shape
        self.img_list = []
        self.density_list = []
        self.corp_label_list = []
        self.data_set_scale = {
            "LL":32,
            "L":64,
            "M":128,
            "s":256,
        }
        self.crop_map(data_set_number)
        
    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        return self.img_list[index], self.density_list[index], self.corp_label_list[index]

    def __len__(self):
        return len(self.img_list)
    
    def get_train_dataset(self, train_size = 0.8, shuffle = False):
        total = len(self.img_list)
        index_list = [ i for i in range(total) ]
        if(shuffle):
            np.random.shuffle(index_list)
            
        train_index_end = int(train_size * total)
        
        index_step = int((total - train_index_end)/2)
        val_index_end = index_step + train_index_end
        test_index_start = val_index_end
        
        return {
            "train_data_x":self.img_list[:train_index_end],
            "train_data_y":{
                "cord":self.corp_label_list[:train_index_end],
                "density":self.density_list[:train_index_end],
            },
            "test_data_x":self.img_list[train_index_end:val_index_end],
            "test_data_y":{
                "cord":self.corp_label_list[train_index_end:val_index_end],
                "density":self.density_list[train_index_end:val_index_end],
            },
            "val_data_x":self.img_list[test_index_start:],
            "val_data_y":{
                "cord":self.corp_label_list[test_index_start:],
                "density":self.density_list[test_index_start:],
            },
            
        }

    @log(TYPE_TEST, "crop map")
    def crop_map(self, data_set_number = "s"):
        data_scale = self.data_set_scale[data_set_number] 
        print("data_scale:",data_scale)
        for raw in range(int((self.img_w - self.img_shape)/data_scale)):
            start_raw, end_raw = raw*data_scale,raw*data_scale + self.img_shape
            for col in range(int((self.img_h - self.img_shape)/data_scale)):
                
                start_col, end_col = col*data_scale, col*data_scale + self.img_shape
                normal_image = copy.deepcopy( self.imgarr_normal[start_raw:end_raw, start_col:end_col] )
                # n_min = np.min(normal_image)
                # n_max = np.max(normal_image)
                # normal_image = (normal_image - n_min)/ (n_max - n_min) 

                if( normal_image.shape != (self.img_shape,self.img_shape)):
                    print("error from shape: ",(end_raw-start_raw,end_col-start_col),",", start_raw,end_raw, start_col,end_col  )

                
                
                mask = np.ones((self.img_shape,self.img_shape), dtype=np.uint8)
                mask[ self.density[start_raw:end_raw, start_col:end_col] <0.02] = 0
                n = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                temp_label_list = []
                for label in self.label_list:
                    if(start_col <= label['x'] and  label['x'] < end_col and start_raw <= label['y'] and label['y'] < end_raw ):
                        label['x_offset'] =  col*data_scale 
                        label['y_offset'] =    raw*data_scale
                        label["x_"] = label['x']  - col*data_scale
                        label['y_'] = label['y'] -  raw*data_scale
                        label['y_normal'] = label['y_']/self.img_shape 
                        label['x_normal'] = label['x_']/self.img_shape
                        temp_label_list.append(label)
                
                if(len(n[0]) != len(temp_label_list)):
                    temp_label_list = []
                    for label in self.label_list:
                        if(start_col - (self.img_shape/16 ) <= label['x'] and  label['x'] < end_col + (self.img_shape/16) and start_raw - (self.img_shape/16) <= label['y'] and label['y'] < end_raw + (self.img_shape/16) ):
                            label['x_offset'] =  col*data_scale
                            label["x_"] = label['x']  - col*data_scale

                            label['y_offset'] =    raw*data_scale
                            label['y_'] = label['y'] -  raw*data_scale

                            if( label["x_"] > self.img_shape):
                                label["x_"] = self.img_shape
                            if( label["y_"] > self.img_shape ):
                                label["y_"] = self.img_shape
                            if(label['x_'] < 0):
                                label['x_'] = 0
                            if(label['y_']<0):
                                label['y_'] = 0

                            label['y_normal'] = label['y_']/self.img_shape
                            label['x_normal'] = label['x_']/self.img_shape
                            temp_label_list.append(label)
                if(len(n[0]) == len(temp_label_list)):
                    self.corp_label_list.append( copy.deepcopy(temp_label_list))
                    self.img_list.append(normal_image)

                    self.density_list.append(self.density[start_raw:end_raw, start_col:end_col])
        
                
                
    
    @log(TYPE_TEST, "read astropy map")
    def read_map(self, file_name):
        imgarr = fits.getdata(file_name)
        self.img_w, self.img_h= imgarr.shape
        if(TYPE_TEST == "DEBUG"):
            print("get fits")
            print("w,h:", self.img_w, self.img_h)
            print("mean:", np.mean(imgarr))
            print("std:", np.std(imgarr))
        imgarr_normal = (imgarr - np.mean(imgarr))/np.std(imgarr)
        self.imgarr = imgarr
        self.imgarr_normal = imgarr_normal
        
    @log(TYPE_TEST, "read density map")
    def read_density(self, file_name):
        with h5py.File(file_name, 'r') as hf:
            self.density = copy.deepcopy(hf['density'][:][:])
        if(TYPE_TEST == "DEBUG"):
            print(self.density.shape)
            
    
    @log(TYPE_TEST, "read cat label")
    def read_label(self, file_name):
        data_list = []
        file_to_read = open(file_name, 'r') 
        lines = file_to_read.readline()
        lines.split('\t')
        index = 1
        while True:
            value_dict = {}
            lines = file_to_read.readline() # 整行读取数据
            value_list = lines.split('\t')
            if(len(value_list) < 11):
                break
            value_dict['x'] = float(value_list[5])
            value_dict['y'] = float(value_list[6])
            value_dict['FWHM'] = float(value_list[7])
            value_dict['MAG'] = float(value_list[3])
            value_dict['index'] = index
            index += 1
            data_list.append(value_dict)
            if not lines:
                break
        file_to_read.close()
        if(TYPE_TEST == "DEBUG"):
            print("data shape:",len(data_list))
        self.label_list = data_list

if __name__ == "__main__":
    a = DataGenerate(data_set_number = "L")
    print(len(a))