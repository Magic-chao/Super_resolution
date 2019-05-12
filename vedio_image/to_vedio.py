import os

#
# " y4mtobmp: ffmpeg -i xx.y4m -vsync 0 xx%3d.bmp -y"
#    "bmptoy4m: ffmpeg -i xx%3d.bmp  -pix_fmt yuv420p  -vsync 0 xx.y4m -y"
#    "bmptoy4m: ffmpeg -i {0}  -pix_fmt yuv420p  -vsync 0 {1} -y"
import time


def to_vedio():
    path1 = "E:/ali_uku/round1_train_result\SR_H"
    path_list = os.listdir(path1)
    for i in path_list:
        # time.sleep(3)
        path = os.path.join(path1, i)+ "/%3d.bmp"
        out_name = "E:/ali_uku/round1_train_result" + "/" + i + ".y4m"
        # str ="ffmpeg -pattern_type glob -i {0} -c:v libx264 -vf fps=25 -pix_fmt {1}".format(path1, out_name)
        # ffmpeg -f image2 -i /home/ttwang/images/image%d.jpg  -vcodec libx264  tt.mp4
        str = "ffmpeg -i {0}  -pix_fmt yuv420p  -vsync 0 {1} -y".format(path, out_name)
        # "    ffmpeg -i xx%3d.bmp  -pix_fmt yuv420p  -vsync 0 xx.y4m -y"
        os.system(str)
        print(i)
        # "ffmpeg -f -r 25 image2 -i {0} -vcodec libx264 {1}"
        # "ffmpeg -loop 1 -f image2 -i E:/ali_uku/round1_train_label\youku_00000_00049_h_GT_pic\Youku_00000_h_GT/*.jpg -vcodec libx264 -r 25 -t 4 test.mp4"


def go():
    path = "E:/ali_uku/round1_train_result\SR_H/49\%4d.bmp"
    # path_list =os.listdir(path1)
    # path = path1 + "/2" + "/%5d" + ".bmp"
    out_name = "E:/ali_uku/round1_train_result/49.y4m" #+ "/" + "2" + ".y4m"
    str = "ffmpeg -i E:/ali_uku/round1_train_result\SR_H/49\%4d.bmp  -pix_fmt yuv420p  -vsync 0 E:/ali_uku/round1_train_result/49.y4m -y".format(path, out_name)

    os.system(str)
    #ffmpeg -i {0}  -pix_fmt yuv420p  -vsync 0 {1} -y

def to_100file():
    path_pic = "E:/ali_uku/round1_train_result/sr"
    out = "E:/ali_uku/round1_train_result/SR_H"
    list_pic = os.listdir(path_pic)
    for i, (name) in enumerate(list_pic):
        file = str(i // 100)
        if not os.path.exists(os.path.join(out, file)):
            os.mkdir(os.path.join(out, file))

        im = cv2.imread(os.path.join(path_pic, name))
        cv2.imwrite(os.path.join(out, file) + "/" +"%03d"%(i-100*int(file)) +'.bmp', im)
        print(i)


import cv2
import os

if __name__ == "__main__":
    to_vedio()
