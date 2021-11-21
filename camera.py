import cv2
import os
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
filelen = len(os.listdir('./1_train'))
num = filelen + 1
while cv2.waitKey(33) != ord('q'):
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    # press space key to start recording
    
    if cv2.waitKey(33) == ord('s'):
        cv2.imwrite('./1_train/'+str(num)+'.png',frame)
        num +=1
capture.release()
cv2.destroyAllWindows()
