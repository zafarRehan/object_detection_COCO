import cv2


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = 'labels.txt'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')
    
    
model.setInputSize(320, 320) #greater this value better the reults tune it for best output
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

    
img = cv2.imread('your image path')

classIndex, confidence, bbox = model.detect(img, confThreshold=0.5) #tune confThreshold for best results


font = cv2.FONT_HERSHEY_PLAIN

for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd-1], (boxes[0] + 10, boxes[1] + 40), font, fontScale = 3, color=(0, 255, 0), thickness=3)
    
    
cv2.imshow('result', img)
cv2.waitKey(0)

cv2.imwrite('result.png', img)

