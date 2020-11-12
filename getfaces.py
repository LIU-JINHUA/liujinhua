import cv2
import sys
import os.path
from glob import glob

def detect(filename,cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename) #读取的图片为BGR格式
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) #使直方图像素点分布均匀

    faces = cascade.detectMultiScale(
        gray,
        # detector options
        scaleFactor = 1.1, #每次缩小图像比例
        minNeighbors = 3,
        minSize = (32,32)
    )

    for i,(x,y,w,h) in enumerate(faces):
        face = image[y: y+h, x:x+w, :]
        face = cv2.resize(face,(64,64))
        save_filename = '%s.jpg' % (os.path.basename(filename).split('.')[0])
        cv2.imwrite("faces/"+save_filename,face)

if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob('imgs/*.jpg') #获得指定目录下所有jpg文件
    for filename in file_list:
        detect(filename)
