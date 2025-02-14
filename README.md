# chinese_licence_plate_generator
A tool for generating Chinese license plate dataset for plate detecting
 
### To run

```sh
python fake_plate_generator.py
```

### Run with arguments to indicate number of images wanted (Default is 1000)
#### '--img_dir', '-id' : put generated image directory
#### '--num_imgs', '-ni' : put number of generated images wanted
#### '--resample', '-r' : put resample value such that it shrinks image by ImageSize/resample and re-enlarges it
#### '--gaussian', '-g' : put range of gaussian blur
#### '--noise', '-n' : put range of noise strength
#### '--rank_blur', '-rb':  rank blur range, odd number only 
```sh
python fake_plate_generator.py -id `pwd`/output/ -ni 100 -r 3 -g 4 -n 5
 python fake_plate_generator.py -id  /ssd/wfei/data/synthetic/cn -ni 100 -r 1 -g 2  -n 2 -rb 1 
```

#### Images are stored in the folder test_plate
#### Background images are randomly selected from the folder "/fake_resources/plate_background_use"
#### Numbers and Lettters are randomly selected from the folder "/fake_resources/numbers" and "/fake_resources/numbers respectively"
