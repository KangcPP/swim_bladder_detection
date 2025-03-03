import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
from scipy import stats
import os
import re
import pickle
import imutils

def init_condition(avis):
    capture = cv.VideoCapture(avis)
    _, frame1 = capture.read()
    num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    # Create mask
    hsv = np.zeros_like(frame1)
    # Make image saturation to a maximum value
    hsv[..., 1] = 255
    return capture, num_frames,hsv

move=[]

def calculate_movement(capture, num_frames, hsv):
    mag1 = []
    ang1 = []
    strike_frame = []
    mag_final = []
    arg_final = []
    mseval=[]
    mseval1 = []


    for i in range(1, num_frames):
      if i < (num_frames) - 1 :
        pre = capture.set(1, i-1)
        _,pre = capture.read()
        pre_final = cv.cvtColor(pre, cv.COLOR_BGR2GRAY)

        cur = capture.set(1, i)
        _,cur=capture.read()
        cur_final = cv.cvtColor(cur, cv.COLOR_BGR2GRAY)

        h, w = cur_final.shape
        #diff= ((cur_final.astype("float") - pre_final.astype("float")) )
        diff = cv.subtract(cur_final, pre_final)
        # diff = cur - pre
        #err1 = np.sum((diff))/(float(h * w))
        err1 = np.mean(diff) / (float(h * w))
        #mse=ssim(cur_final,pre_final)
        err = np.sum((cur_final.astype("float") - pre_final.astype("float")) ** 2)
        #err /= float(imageA.shape[0] * imageA.shape[1])
        mse = err / (float(h * w))
        mse1 = err1  #/ (float(h * w))
        mseval.append(mse)
        theta = mse
        thresh1=theta
        mseval1.append(mse1)

    res=[]
    stdtest= np.mean(mseval1) + 6000 * np.std(mseval1)
    print(stdtest)
    stdtest1 = np.mean(mseval) + 3.25 * np.std(mseval)
    #print(stdtest1)
    mean=np.mean(mseval1)
    std=np.std(mseval1)
    #print(stdtest)
    #print(2*stdtest)
    #print(3*stdtest)
    # plt.figure(figsize=(14, 10))
    # ax = sns.kdeplot(mseval1, shade=True, color='crimson')
    # ax.axvline(x=mean,color='crimson',linestyle='--')
    # plt.axvline(x=mean+1*std, color='crimson',linestyle='--')
    # plt.axvline(x=mean+2*std, color='crimson',linestyle='--')
    # plt.axvline(x=mean + 3 *std, color='crimson',linestyle='--')
    # plt.show()
    #
    #
    from scipy.stats import norm

    # N = 10
    # for i in [1, 2, 3, 4]:
    #     x1 = np.linspace(mean - i * std, mean - (i - 1) * std, N)
    #     x2 = np.linspace(mean - (i - 1) * std, mean + (i - 1) * std, N)
    #     x3 = np.linspace(mean + (i - 1) * std, mean + i * std, N)
    #     x = np.concatenate((x1, x2, x3))
    #     x = np.where((mean - (i - 1) * std < x) & (x < mean + (i - 1) * std), np.nan, x)
    #     y = norm.pdf(x, mean, std)
    #     ax.fill_between(x, y, alpha=0.5)

    # plt.xlabel("Grey scale difference between consecutive frames", fontsize=33)
    # plt.ylabel("Density", fontsize=33)
    #plt.xticks(ticks=range(0, 10))
    #plt.grid()
    # ax.set_ylabel('Rate (%)',fontsize=33)
    #plt.yticks(fontsize=33)
    # ax.set_xlabel('',fontsize=33)
    # ax.set_xlim(-0.5,3.20-0.5)
    # plt.xticks(fontsize=33)
    # plt.rcParams["font.family"] = "Arial"
    #plt.savefig('C:\\Users\\user\\OneDrive - HKUST Connect\\Desktop\\test.jpeg', dpi='figure')
    #plt.show()
    #plt.hist(mseval)
    #plt.show()
    for a in range(0,len(mseval)):
        if mseval[a] > stdtest1:
            res.append(a)

    #print(res)
    # sns.distplot(mseval)
    #print("Hello")
    #print(res)
    # print(len(res))
    # plt.show()
    # plt.close()
    directions_map = np.zeros([10, 5])
    for k in range(0, len(res)):
        #print(res[k])
        param = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 50,
            'iterations': 1,
            'poly_n': 5,
            'poly_sigma': 1.5,
            #'flags': cv.OPTFLOW_FARNEBACK_GAUSSIAN,
            'flags': cv.OPTFLOW_LK_GET_MIN_EIGENVALS
        }
        #pre = capture.set(1,1)
        #_, pre = capture.read()
        #pre_final = cv.cvtColor(pre, cv.COLOR_BGR2GRAY)
        #pre_final = cv.GaussianBlur(pre_final, (21, 21), 0)
        if res[k] < (num_frames) - 1:
            # print(res[k]-1)
            # print(res[k])
            pre = capture.set(1, res[k-1])
            _, pre = capture.read()
            pre_final = cv.cvtColor(pre, cv.COLOR_BGR2GRAY)
            # pre_final = cv.GaussianBlur(pre_final, (21, 21), 0)

            cur = capture.set(1, res[k])
            _, cur = capture.read()

            cur_final = cv.cvtColor(cur, cv.COLOR_BGR2GRAY)
            #cur_final = cv.GaussianBlur(cur_final, (21, 21), 0)


            flow = cv.calcOpticalFlowFarneback(pre_final,cur_final, None, **param)
            mag1, ang1 = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

            move_sense = ang1[mag1 > stdtest]
            #print(res[k])

            move_mode = (stats.mode(move_sense)[0])
            #plt.hist(move_sense)
            #plt.show()
            #print(move_mode)
            if move_mode != 0:

               move_mode = float(move_mode)
            else:
                    move_mode = 0.0


            #print(move_mode)

            # if 10 < move_mode <= 100:
            #     loc=0
            # elif 100 < move_mode <= 190:
            #     loc=1
            # elif 190 < move_mode <= 280:
            #     loc=2
            # elif 280 < move_mode or move_mode < 10:
            #     loc=3
            # else:
            #    loc=4

            # if 60 < move_mode <= 120:
            #      loc=0
            # elif 150 < move_mode <= 210:
            #      loc=1
            # elif 240 < move_mode <= 300:
            #      loc=2
            # elif 330 < move_mode or move_mode < 30:
            #      loc=3
            # else:
            #      loc=4

            if 70 < move_mode <= 100:
                 loc=0
            elif 170 < move_mode <= 190:
                 loc=1
            elif 260 < move_mode <= 280:
                 loc=2
            elif 350 < move_mode or move_mode < 10:
                 loc=3
            else:
                 loc=4

            ang_180 = ang1 / 2
            #print(loc)
            #loc = directions_map.mean(axis=0).argmax()
            if loc == 0:
                text = 'Moving down'
            elif loc == 1:
                text = 'Moving to the right'
            elif loc == 2:
                text = 'Moving up'
            elif loc == 3:
                text = 'Moving to the left'
            else:
                text = 'No Movement'

            #move.append(text)
            if loc == 2 : #or loc == 0:
                    # print(res[k])
                    # print(loc)
                    # print(thresh1)

                    strike_frame.append(res[k])
                    mag_final.append(np.mean(mag1))
                    arg_final.append(move_mode)

            else:
                    #print(res[k])
                    arg_final.append(0)
                    mag_final.append(0)

        hsv[:, :, 0] = ang_180
        hsv[:, :, 2] = cv.normalize(mag1, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        if text == 'Moving up':

            colr = (0, 0, 255)
        else:
            colr = (0, 0, 0)

        cv.putText(cur, 'Frame Number ' + ': ' + str(res[k]), (30, 50), cv.FONT_HERSHEY_COMPLEX, cur.shape[1] / 500,
                   colr, 2)
        cv.putText(cur, text, (30, 80), cv.FONT_HERSHEY_COMPLEX, cur.shape[1] / 500, colr, 2)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        cv.imshow('Frame', cur)


    cv.destroyAllWindows()
    return mag_final, arg_final, strike_frame

def plot(mag_final, arg_final, strike_frame, filename):
    print(strike_frame)
    #print(len(move))
    #print(move)
    with open(filename +".pickle", "wb") as f:
        pickle.dump(strike_frame, f, protocol=2)
        pickle.dump(mag_final, f,protocol=2)
        pickle.dump(arg_final, f,protocol=2)


def main():
        main_dir = "D://testing_data//F6//cropped//"
        directory_contents = os.listdir(main_dir)

        list_folder = main_dir
    # for i in range(0, len(directory_contents)):
    #     list_folder.append(main_dir + '/' + directory_contents[i])

        print(list_folder)
        folder = list_folder
    # for a in range(0, len(list_folder)):
    #     # # READ ALL THE AVIS FILES WITHIN THAT FOLDER
    #     folder = list_folder[a]
    #     print(folder)
        filenames = os.listdir(folder)
        print(filenames)

        avis = [filename for filename in filenames if re.search(".avi",os.path.splitext(filename)[1])]
        for k in range(0,len(avis)):
         if len(avis) != 0:

            capture, num_frames, hsv = init_condition(folder + '\\' + avis[k])
            filename1 = folder + '\\' + avis[k]
            filename = filename1.replace('.avi      ', "")

            print(filename1)
            mag_final, arg_final, strike_frame = calculate_movement(capture, num_frames, hsv)
            plot(mag_final, arg_final, strike_frame, filename)


if __name__ == "__main__":
    main()


