import cv2
import numpy as np
import cv2
import pandas as pd
import os



# Size of your cropped video, here is 300×300
crop_size = 50

def crop_video(input_path, output_path, crop_size):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps,width,height,crop_size)
    # Change your start point here
    crop_x = 258
    crop_y = 147
    #crop_x = 180
    #crop_x= 300
    #crop_y = 250
    # crop_x= 510
    # crop_y = 135

    # output
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  #(*'XVID')  # AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))

    ret, frame = cap.read()

    if ret == True:

    #Start cropping
     while True:
        ret, frame = cap.read()
        if not ret:
            break

        # keep middle areas
        cropped_frame = frame[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        out.write(cropped_frame)


     cap.release()
     out.release()
     cv2.destroyAllWindows()



# Your input path
#input_video_path ="C:\\Users\\user\\Downloads\\for_Biswadeep_sbanalysis\\"
#F2_60__uv_bilateral_2mm_7deg_contrast-0.4
#F3_60__uv_bilateral_4mm_4deg_contrast0.325
#input_video_path ="C:\\Users\\user\\Downloads\\2024-05-17\\2024-05-17\\F3_60__uv_bilateral_2mm(3mm image not clear)_6deg\\\TOPCAMERA\\"
input_video_path ="D://testing_data//F6//"

# Your output path

output_video_path = input_video_path + "cropped//"
if not os.path.exists(output_video_path):
    os.makedirs(output_video_path)

print(output_video_path)

def main():
    for file in os.listdir(input_video_path):
        # Loop through every file in input path
        if ".avi" in file:
            file_name = file.split("\\")[-1]
            result_file_name = os.path.splitext(file_name)[0] + "_cropped.avi"
            output_path=output_video_path+result_file_name
            print(file)
            crop_video(input_video_path+file, output_path, crop_size)

if __name__ == "__main__":
    main()
