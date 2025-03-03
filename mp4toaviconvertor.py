import os
import math

path='D://testing_data//F6//'
for file in os.listdir(path):
    print(path)
    if (file.endswith(".mp4")): #or .avi, .mpeg, whatever.
        filename=path+file
        filename1=filename.replace(".mp4","")
        print(filename1)
        os.system(f'ffmpeg -i {filename} -c:v mjpeg -q:v 1 -vf "format=yuvj420p" -c:a copy  {filename1}.avi')
    else:
        continue