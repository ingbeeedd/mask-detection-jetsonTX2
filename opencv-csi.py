import cv2
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

import time

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/mask_detector.model')

def gstreamer_pipeline(
    capture_width=400,
    capture_height=400,
    display_width=640, 
    display_height=480,
    framerate=1,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def read_cam():
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps {}fps'.format(fps))

        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            cv_time = time.time()
            ret_val, img = cap.read()
            
            h, w = img.shape[:2]

            blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
            facenet.setInput(blob)
            dets = facenet.forward()

            result_img = img.copy()

            print('cv face detect time : {}'.format(time.time() - cv_time))

            model_time = time.time()

            for i in range(dets.shape[2]):
                confidence = dets[0, 0, i, 2]
                if confidence < 0.5:
                    continue

                x1 = int(dets[0, 0, i, 3] * w)
                y1 = int(dets[0, 0, i, 4] * h)
                x2 = int(dets[0, 0, i, 5] * w)
                y2 = int(dets[0, 0, i, 6] * h)
                
                face = img[y1:y2, x1:x2]

                face_input = cv2.resize(face, dsize=(224, 224))
                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
                face_input = preprocess_input(face_input)
                face_input = np.expand_dims(face_input, axis=0)
                
                mask, nomask = model.predict(face_input).squeeze()

                if mask > nomask:
                    color = (0, 255, 0)
                    label = 'Mask %d%%' % (mask * 100)
                else:
                    color = (0, 0, 255)
                    label = 'No Mask %d%%' % (nomask * 100)

                cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
                cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

            print('inference face detect time : {}'.format(time.time() - model_time))

            # out.write(result_img)
            cv2.imshow('result', result_img)
            cv2.imshow("CSI Camera", img)
            # This also acts as
            # keyCode = cv2.waitKey()
            # Stop the program on the ESC key
            
            print('total detect time : {}'.format(time.time() - cv_time))

            #cv2.imshow("CSI Camera", img)
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
            
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == '__main__':
    read_cam()