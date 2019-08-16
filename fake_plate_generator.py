#coding=utf-8
from __future__ import division
import itertools
import math
import os
import random
import sys
import numpy as np
import cv2
import codecs
import json

from img_utils import *
from jittering_methods import *
from parse_args import parse_args

args = parse_args()
fake_resource_dir  = sys.path[0] + "/fake_resource/" 
output_dir = args.img_dir
resample_range = args.resample 
gaussian_range = args.gaussian 
noise_range = args.noise
rank_blur = args.rank_blur
brightness = args.brightness
motion_blur = args.motion_blur
chinese_dir = fake_resource_dir + "/chinese/"
number_dir = fake_resource_dir + "/numbers/" 
letter_dir = fake_resource_dir + "/letters/" 
plate_dir = fake_resource_dir + "/plate_background_use/"
character_y_size = 47
character_y_size_Top = 25
plate_y_size = 90#164

class FakePlateGenerator(): 
    def __init__(self, fake_resource_dir, plate_size):


        self.dst_size = plate_size

        self.chinese_Top = self.load_image_top(chinese_dir, character_y_size_Top)
        self.numbers = self.load_image(number_dir, character_y_size)
        self.letters = self.load_image(letter_dir, character_y_size)
        self.letters_Top = self.load_image_top(letter_dir, character_y_size_Top)
        self.numbers_and_letters = dict(self.numbers, **self.letters)

        #we only use blue plate here
        self.plates, self.plate_x_size = self.load_plate_image(plate_dir, plate_y_size)
    
        for i in self.plates.keys():
            self.plates[i] = cv2.cvtColor(self.plates[i], cv2.COLOR_BGR2BGRA)

        #take "苏A xxxxx" for example

        #position for "苏A"
        self.character_position_x_list_part_1 = [80, 150]  
        #position for "xxxxx"              
        self.character_position_x_list_part_2 = [35, 75, 115, 155, 195]
    
    def get_radom_sample(self, data):
        keys = list(data.keys())
        i = random.randint(0, len(data) - 1)
        key = keys[i]
        value = data[key]

        #注意对矩阵的深拷贝
        return key, value.copy()

    def load_image(self, path, dst_y_size):
        img_list = {}
        current_path = sys.path[0]

        listfile = os.listdir(path)     

        for filename in listfile:
            img = cv2.imread(path + filename, -1)
            
            height, width = img.shape[:2]
            x_size = int(1.4 * int(width*(dst_y_size/float(height))))
            img_scaled = cv2.resize(img, (x_size, dst_y_size), interpolation = cv2.INTER_CUBIC)
            
            img_list[filename[:-4]] = img_scaled

        return img_list
    
    def load_image_top(self, path, dst_y_size):
        img_list = {}
        current_path = sys.path[0]

        listfile = os.listdir(path)     

        for filename in listfile:
            img = cv2.imread(path + filename, -1)
            
            height, width = img.shape[:2]
            x_size = int(3.5 * int(width*(dst_y_size/float(height))))
            img_scaled = cv2.resize(img, (x_size, dst_y_size), interpolation = cv2.INTER_CUBIC)
            
            img_list[filename[:-4]] = img_scaled

        return img_list
    
    def load_plate_image(self, path, dst_y_size):
        img_list = {}
        current_path = sys.path[0]

        listfile = os.listdir(path)     

        for filename in listfile:
            img = cv2.imread(path + filename, -1)
            
            height, width = img.shape[:2]
            # dst_y_size_temp = int(dst_y_size/2)
            x_size = int(width*(dst_y_size/float(height)))+50
            img_scaled = cv2.resize(img, (x_size, dst_y_size), interpolation = cv2.INTER_CUBIC)
            
            img_list[filename[:-4]] = img_scaled

        return img_list, x_size


    def add_character_to_plate_Top(self, character, plate, x):
        h_plate, w_plate = plate.shape[:2]
        h_character, w_character = character.shape[:2]

        start_x = x - int(w_character/2)
        start_y = 6

        a_channel = cv2.split(character)[3]
        ret, mask = cv2.threshold(a_channel, 100, 255, cv2.THRESH_BINARY)

        overlay_img(character, plate, mask, start_x, start_y)
        return start_x, start_y

    def add_character_to_plate(self, character, plate, x):
        h_plate, w_plate = plate.shape[:2]
        h_character, w_character = character.shape[:2]

        start_x = x - int(w_character/2)
        start_y = int((h_plate - h_character)/2) + 13

        a_channel = cv2.split(character)[3]
        ret, mask = cv2.threshold(a_channel, 100, 255, cv2.THRESH_BINARY)

        overlay_img(character, plate, mask, start_x, start_y)
        return start_x, start_y

    def generate_one_plate(self):
        plate_chars = ""
        _, plate_img = self.get_radom_sample(self.plates)
        plate_name = ""
        start_xy= []

        character, img = self.get_radom_sample(self.chinese_Top)
        start_xy.append(self.add_character_to_plate_Top(img, plate_img, self.character_position_x_list_part_1[0]))
        plate_name += "%s"%(character,)
        plate_chars += character

        character, img1 = self.get_radom_sample(self.letters_Top)
        start_xy.append(self.add_character_to_plate_Top(img1, plate_img, self.character_position_x_list_part_1[1]))
        plate_name += "%s"%(character,)
        plate_chars += character

        for i in range(5):
            character, img2 =  self.get_radom_sample(self.numbers_and_letters)
            start_xy.append(self.add_character_to_plate(img2, plate_img, self.character_position_x_list_part_2[i]))
            plate_name += character
            plate_chars += character

        #转换为RBG三通道
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2BGR)
  
        #转换到目标大小
        pt1 = (start_xy[0][0],start_xy[0][1])
        pt2 = (start_xy[1][0] + img1.shape[1], start_xy[1][1] + character_y_size_Top)
        pt3 = (start_xy[2][0], start_xy[2][1])
        pt4 = (start_xy[6][0] + img2.shape[1], start_xy[6][1] + character_y_size)
        # cv2.rectangle(plate_img, pt1,pt2, (0,0,255), 2)
        # cv2.rectangle(plate_img, pt3, pt4,(0,0,255),2)
        # cv2.imshow(' ', plate_img)
        # cv2.waitKey(0)

        plate_img = cv2.resize(plate_img, self.dst_size, interpolation = cv2.INTER_AREA)

        scalex = self.dst_size[0]/self.plate_x_size
        scaley = self.dst_size[1]/plate_y_size

        pts1 = (int(pt1[0] * scalex)-3,int(pt1[1] * scaley)-3) 
        pts2 = (int(pt2[0] * scalex)+3,int(pt2[1] * scaley)+3)
        pts3 = (int(pt3[0] * scalex)-3, int(pt3[1] * scaley)-3)
        pts4 = (int(pt4[0] * scalex)+3, int(pt4[1] * scaley)+3)

        # cv2.rectangle(plate_img, pts1,pts2, (0,0,255), 2)
        # cv2.rectangle(plate_img, pts3, pts4,(0,0,255),2)
        # cv2.imshow(' ', plate_img)
        # cv2.waitKey(0)
        box = ((pts1[0],pts1[1], pts2[0] - pts1[0], pts2[1] - pts1[1]),(pts3[0],pts3[1],pts4[0]-pts3[0],pts4[1]-pts3[1]))
        return plate_img, plate_name, plate_chars, box

def write_to_txt(fo,img_name, plate_characters):
    plate_characters.decode('utf8')
    plate_label = '|' + '|'.join(plate_characters.decode('utf8')) + '|'
    print plate_label
    line = img_name.decode('utf8') + ';' + plate_label.upper() + '\n'
    print line
    fo.write("%s" % line)

def json_generator(json_file, fname, plate_chars,box,json_data):
    i=0
    plate_anno =[]
    plate_anno.append(
        {"box": box,
        "text": plate_chars.decode('utf-8'),}
    )
    if fname not in json_data:
        json_data[fname.split('/')[len(fname.split('/'))-1]] = plate_anno


if __name__ == "__main__":
    img_size = (300, 180)
    json_data = {}
    json_file = output_dir + "/ocr_label.json"
    reset_folder(output_dir)
    numImgs = args.num_imgs
    fo = codecs.open(output_dir + 'labels.txt', "w", encoding='utf-8')
    for i in range(0, numImgs):
        fake_plate_generator= FakePlateGenerator(fake_resource_dir, img_size)
        plate, plate_name, plate_chars, box = fake_plate_generator.generate_one_plate()
        # #plate = underline(plate)
        # plate = jittering_color(plate)
        # plate = add_noise(plate,noise_range)
        # plate = jittering_blur(plate,gaussian_range)
        # plate = resample(plate, resample_range)
        # plate = jittering_scale(plate)
        # # plate = perspectiveTransform(plate)
        # plate = random_rank_blur(plate,rank_blur)
        # plate = random_motion_blur(plate,motion_blur)
        # plate = random_brightness(plate, brightness)
        file_name = save_random_img(output_dir,plate_chars.upper(), plate)
        write_to_txt(fo,file_name,plate_chars)
        json_generator(json_file, file_name,plate_chars.upper(), box,json_data)
    with codecs.open(json_file, 'w',encoding='utf-8') as f:
        f.write(json.dumps(json_data,indent=4,ensure_ascii=False))
