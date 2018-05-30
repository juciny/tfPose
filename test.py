import cv2
import numpy as np

if __name__=='__main__':
    video="child.mp4"
    cap=cv2.VideoCapture(video)

    logo = cv2.imread("F:/jupyterNotebook/tf-pose-estimation-master/photo/chibang03.png", -1)
    rows, cols, channels = logo.shape
    weight = 100
    height = 100

    # 缩小图像
    size = (int(weight), int(height))
    logo = cv2.resize(logo, size, interpolation=cv2.INTER_AREA)

    src_channels = cv2.split(logo)
    b, g, r, a = cv2.split(logo)

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        ret_val, frame= cap.read()

        dst_channels=cv2.split(frame)

        #frame[0:rows,0:cols]=logo

        for i in range(3):
            dst_channels[i][100:weight+100,10:height+10] = dst_channels[i][100:weight+100,10:height+10] * (255.0 - a) / 255
            dst_channels[i][100:weight+100,10:height+10] += np.array(src_channels[i] * (a / 255), dtype=np.uint8)
        cv2.imshow('tf-pose-estimation result', cv2.merge(dst_channels))
        k=cv2.waitKey(10)
        if k&0xff== 27:
            break
    cap.release()
    cv2.destroyAllWindows()