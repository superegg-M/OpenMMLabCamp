import cv2

img=cv2.imread("/home/superegg/PycharmProjects/OpenMMLab/task1_MMPose/2023/0524/mmpose/outputs/B1_HRNet_1/multi-person.jpeg")
cv2.namedWindow("img",0)
cv2.resizeWindow("img",1080,720)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

video=cv2.VideoCapture("/home/superegg/PycharmProjects/OpenMMLab/task1_MMPose/2023/0524/mmpose/outputs/B1_RTM_2/cxk.mp4")
while(video.isOpened()):
    retval,image=video.read()
    # cv2.namedWindow("Video",0)
    # cv2.resizeWindow("Video",420,300)
    if retval==True:
        cv2.imshow("Video",image)
    else:
        break
    key=cv2.waitKey(1)
    if key==27:
        break
video.release()
cv2.destroyAllWindows()
