import cv2
import sys

cpt=0
vidStream1=cv2.VideoCapture(1)
vidStream=cv2.VideoCapture(0)
while True:
    ret1,frame1=vidStream.read()
    ret, frame = vidStream.read()

    cv2.imshow("test frame1", frame)
    cv2.imshow("test frame", frame1)
    cv2.imshow("test frame2", frame1)
    cv2.imshow("test frame3", frame1)


    if cv2.waitKey(10)==ord('q'):
        break