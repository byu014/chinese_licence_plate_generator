#coding=utf-8
import itertools
import math
import os
import random
import sys
import numpy as np
import cv2
import codecs

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

class FakePlateGenerator():
    def __init__(self, fake_resource_dir, plate_size):

        character_y_size = 110
        plate_y_size = 164

        self.dst_size = plate_size

        self.chinese = self.load_image(chinese_dir, character_y_size)
        self.numbers = self.load_image(number_dir, character_y_size)
        self.letters = self.load_image(letter_dir, character_y_size)

        self.numbers_and_letters = dict(self.numbers, **self.letters)

        #we only use blue plate here
        self.plates = self.load_image(plate_dir, plate_y_size)
    
        for i in self.plates.keys():
            self.plates[i] = cv2.cvtColor(self.plates[i], cv2.COLOR_BGR2BGRA)

        #take "苏A xxxxx" for example

        #position for "苏A"
        self.character_position_x_list_part_1 = [43, 111]  
        #position for "xxxxx"              
        self.character_position_x_list_part_2 = [205, 269, 330, 395, 464]
    
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
            x_size = int(width*(dst_y_size/float(height)))
            img_scaled = cv2.resize(img, (x_size, dst_y_size), interpolation = cv2.INTER_CUBIC)
            
            img_list[filename[:-4]] = img_scaled

        return img_list

    def add_character_to_plate(self, character, plate, x):
        h_plate, w_plate = plate.shape[:2]
        h_character, w_character = character.shape[:2]

        start_x = x - int(w_character/2)
        start_y = int((h_plate - h_character)/2)

        a_channel = cv2.split(character)[3]
        ret, mask = cv2.threshold(a_channel, 100, 255, cv2.THRESH_BINARY)

        overlay_img(character, plate, mask, start_x, start_y)

    def generate_one_plate(self):
        plate_chars = ""
        _, plate_img = self.get_radom_sample(self.plates)
        plate_name = ""
    
        character, img = self.get_radom_sample(self.chinese)
        self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_1[0])
        plate_name += "%s"%(character,)
        plate_chars += character

        character, img = self.get_radom_sample(self.letters)
        self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_1[1])
        plate_name += "%s"%(character,)
        plate_chars += character

        for i in range(5):
            character, img =  self.get_radom_sample(self.numbers_and_letters)
            self.add_character_to_plate(img, plate_img, self.character_position_x_list_part_2[i])
            plate_name += character
            plate_chars += character

        #转换为RBG三通道
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2BGR)
  
        #转换到目标大小
        plate_img = cv2.resize(plate_img, self.dst_size, interpolation = cv2.INTER_AREA)

        return plate_img, plate_name, plate_chars

def write_to_txt(fo,img_name, plate_characters):
    plate_characters.decode('utf8')
    plate_label = '|' + '|'.join(plate_characters.decode('utf8')) + '|'
    print plate_label
    line = img_name.decode('utf8') + ';' + plate_label.upper() + '\n'
    print line
    fo.write("%s" % line)

if __name__ == "__main__":
    img_size = (300, 90)

    reset_folder(output_dir)
    numImgs = args.num_imgs
    fo = codecs.open(output_dir + 'labels.txt', "w", encoding='utf-8')
    for i in range(0, numImgs):
        fake_plate_generator = FakePlateGenerator(fake_resource_dir, img_size)
        plate, plate_name, plate_chars = fake_plate_generator.generate_one_plate()
        #plate = underline(plate)
        plate = jittering_color(plate)
        plate = add_noise(plate,noise_range)
        plate = jittering_blur(plate,gaussian_range)
        plate = resample(plate, resample_range)
        plate = jittering_scale(plate)
        # plate = perspectiveTransform(plate)
        plate = random_rank_blur(plate,rank_blur)
        plate = random_motion_blur(plate,motion_blur)
        plate = random_brightness(plate, brightness)
        file_name = save_random_img(output_dir,plate_chars.upper(), plate)
        write_to_txt(fo,file_name,plate_chars)
