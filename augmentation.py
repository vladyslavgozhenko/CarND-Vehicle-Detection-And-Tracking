import glob
import random
import matplotlib.image as mpimg
import cv2


#extracts pathes to images
def read_files_names(path_in):
    cars = []
    images = glob.glob(path_in+'/*.png')
    for image in images:
        cars.append(image)
    return cars

#create augmenated data
def augmenation(files, path_out, iter_num=1, new_size = (64,64), ramdom_center=0.2):
    i = 0
    for file in files:
        image = mpimg.imread(file)
        resized_image = cv2.resize(image, new_size)
        mpimg.imsave(path_out + '/image_' + str(i) + '.png', resized_image)
        i+=1
        for n in range(iter_num):
            rand_x = random.randint(int(image.shape[1]*(1-ramdom_center)/2),int(image.shape[1]*(1+ramdom_center)/2))
            rand_y = random.randint(int(image.shape[0]*(1-ramdom_center)/2),int(image.shape[0]*(1+ramdom_center)/2))
            if rand_x<=image.shape[1]//2:
                size_x_min = 0
                size_x_max = rand_x*2
            else:
                size_x_min = rand_x - image.shape[1]//2
                size_x_max = image.shape[1] - rand_x
            if rand_y<=image.shape[0]//2:
                size_y_min = 0
                size_y_max = rand_y*2
            else:
                size_y_min = rand_y - image.shape[0]//2
                size_y_max = image.shape[0] - rand_y
            new_image = image[size_y_min:size_y_max, size_x_min:size_x_max, :]
            resized_image = cv2.resize(new_image, new_size)
            mpimg.imsave(path_out + '/image_' + str(i) + '.png', resized_image)
            i+=1



path_in = 'tmp2'
path_out = 'vehicles/tmp2'

ramdom_center = 0.25#deviation of the center of new image from old image center
iter_num = 110 #how many samples generate from real sample
new_size = (64,64) #size of each of generated images

files = read_files_names(path_in)
augmenation(files, path_out, iter_num, new_size, ramdom_center)