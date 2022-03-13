# Object Detection

This repository uses pre-trained MobileNet-SSD v3 model for Object Detection
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

The model uses COCO dataset for training
https://cocodataset.org/#overview

The COCO dataset consists of 80 classes of images (<a href="/labels.txt">View all</a>) so the objects detected would belong from one of these classes only.

You can download the test_video from: <br/>
https://drive.google.com/file/d/1EPngtG-X4NCOoadeo1Fk556Yhejuvp5C/view?usp=sharing<br/>
or you can use any video of your choice
<h2> Required Libraries </h2>

    pip install opencv-python
    
</br>
<h2> Usage </h2>
This repository has 2 files: <br/>
1. detection_code_image.py <br/>
2. detection_code_video.py <br/>

<h3> Image </h3>

Change the 'your image path' in detection_code_image.py at line:23 to the path of your image to detect objects in your Image

<h4> Example Input Image: </h4>
<img src="/test_image.png" width=500>
<h4> Example Output Image: </h4>
<img src="/test_output.png" width=500>

<h3> Video </h3>

Change the 'your image path' in detection_code_video.py at line: 21 to the path of your video to detect objects in your Video

<h4> Example Input Video: </h4>
<a href="https://drive.google.com/file/d/1EPngtG-X4NCOoadeo1Fk556Yhejuvp5C/view?usp=sharing"><img src="/input_video_thumb.png" width=500></a>

<h4> Example Output Video: </h4>
<a href="https://drive.google.com/file/d/1y52-GFJHX28FRxgSJZVlkGvQsnY8ML75/view?usp=sharing"><img src="/output_thumb.png" width=500></a>
