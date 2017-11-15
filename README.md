## Vehicle Detection

<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/ZoneBetweenLinesBig.png" width="480" alt="lane lines" />
</p>

In this project, the goal is to write a software pipeline to detect vehicles in a video. Code will be partially reused from the previous project "Advanced Lane Finding" (the first 724 rows of code).

The steps of this project are the following:

* Analysis of the input data/images.
* Extraction  the spatial features of the images.
* Extraction of the color features of the images.
* Extraction of Histogram of Oriented Gradients(HOG) features of the images.
* Selection of the color spaces such as LUV or HLS.
* Stack and scale of the different features of the images.
* Normalization of the features and randomization of a selection for training and testing.
* Training of the linear classifier.
* Implementation of a sliding-window technique and use it with the trained classifier to search for vehicles in images.
* Run of the chosen pipeline on the video stream.
* Analysis of the results and possible improvements.

### Pipeline
---
Since there is mostly processing of visual information in the project, I will demonstrate each step of the pipeline with a picture. To see exact code for every transformation, please check the file [carnd_advanced_lines.py](carnd_advanced_lines.py). To be sure, that the chosen algorithms perform well on test images, all test images will be processed and visualised.

---
#### Analysis of the input data/images
---
The labeled data used for training the classifier was download from here:
[vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.
There is a count of 8792  cars and 8968  non-cars
of size:  (64, 64, 3)  and data type: float32
The data set seems to be balanced well.
I glance over the folders with the images for cars/notcars and didn't see any mismatch between labels and images.The data can be used to train a classifier.
Each images of a car/not a car is a image of 64 by 64 pixels with 3 RGB channels.
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/cv2.drawChessboardCorners.jpg" width="480" alt="lane lines" />
</p>


#### Extraction of the different features of images

---
The simplest way would be to do template matching between the images in our training set and the test images. The problem with this method, that it is not possible to have templates for all possible situations (weather conditions, day time, car color etc.), therefore I won't consider this technique here.
In my pipeline I used 3 techniques:
1) Histogram of colors (histogram of pixel intensity (color histograms) is used as features):
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/abs_sobel_thresh.jpg" width="480" alt="lane lines" />
</p>
32 histogram color bins were used with bins range (0,256). For the 8-bit images bigger values than 32 won't bring any improvement, but will consume more memory. With lower number of color bins will decrease a number of significant features in the output.
2) Spatial Binning of Color (raw pixel values are used to get a feature vector in searching for cars):
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/abs_sobel_thresh.jpg" width="480" alt="lane lines" />
</p>
Spatial binning dimensions were 32 by 32. Bigger dimensions required much more memory than smaller dimensions, small dimensions do not provide enough significant features to separate car from notcars. 32 by 32 is the optimum for my pipeline and hardware.
3) Histogram of Oriented Gradients(HOG) features of the images (shows gradients of values in images).
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/abs_sobel_thresh.jpg" width="480" alt="lane lines" />
</p>
For the HOG following parameters were used:
orient = 9 . Orientation binning: each pixel within the cell casts a weighted vote for an orientation-based histogram channel based on the values found in the gradient computation. Dalal and Triggs found that unsigned gradients used in conjunction with 9 histogram channels performed best in their human detection experiments, therefore I will stick with 9 histogram channels for the car detection task.
pix_per_cell = 8  Image is divided into small sub-images: “cells” 8 by 8 pixels and accumulation of a histogram of edge orientation withing each cell is done.
cell_per_block = 2 To provide better illumination invariance (lighting, shadows,
etc.) normalization of the cells across larger regions incorporating
multiple cells: “blocks” is done.
The values of HOG parameters were chosen after some experiments with them and recommendation from different sources (e.g. Dalal and Triggs).

After calculating the features I stuck them vertically for further use:
X = np.vstack((car_features, notcar_features)).astype(np.float64)

Color space/conversion RGB2YCrCb was used in the pipeline (I tried some other color spaces, but with RGB2YCrCb I achieved faster better results and kept using it further in the pipeline).
---
#### Features normalization and training the classifier
---
 Now I can train a classifier, but first, as in any machine learning application, the data has to be normalized. Python's sklearn package provides the StandardScaler() method to scale the feature vectors to zero mean and unit variance.

from sklearn.preprocessing import StandardScaler


There a lot of classifiers in sklean.svm (e.g. LinearSVC, NuSVC, SVR etc.). I tried several of them and linear SVC showed the optimal speed/accuracy result on my hardware:

23.72 Seconds to train SVC
Test Accuracy of SVC =  0.982
My SVC predicts:  [ 1.  1.  1.  0.  0.  0.  1.  1.  0.  1.]
For these 10 labels:  [ 1.  1.  1.  0.  0.  0.  1.  0.  0.  1.]
0.00154 Seconds to predict 10 labels with SVC

The data was split in training and validation set. Better than 98% accuracy was achieved with the linear classifier on the validation set.

 Here are results of applying trained classifier to the test images. The heatmap shows sliding windows where the classifier detected cars.

 <img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/abs_sobel_thresh.jpg" width="480" alt="lane lines" />
 </p>

Having pipeline, that works on the individual frames, it was not difficult to apply the pipeline on individual frames of videos. The program analyzes/remembers several video frames and then sum together using bitwise_and operation. I there were any false detections, the probability is high, that those false detection won't happen in a sequence of frames. Otherwise, if there is a car, it will be detected on the most adjacent video frames.
heat = cv2.bitwise_and(heatmap_global[i], heat)
The pipeline did work well on test project video: there are not a lot false positive detections on the video.
The are some issues with detection of the white car. I think, there is not many white cars in the training set and there are not many cars on the furthest right lane, so the classifier underfits. Bigger data set is necessary.


#### Possible improvements
---
* To improve true detection rate bigger training sets are necessary.
* Performance of other classifiers should be examined more thoughtfully.
* To predict trajectory of detected cars their speed and direction can be estimated from the frames with true detection. The known curvature of the road can help to predict steering angles of other cars.
* Combination with other techniques/hardware can improve the detection rate of the system. E.g. LIDAR can identify where obstacles are and the optical detection system can analyze only that direction.
