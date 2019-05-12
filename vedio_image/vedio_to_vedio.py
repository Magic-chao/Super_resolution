import os

def y4m_yuv():
    path="E:/ali_uku/round1_train_input\youku_00000_00049_l"
    path_list =os.listdir(path)
    for name in path_list:
        out_path ="E:/ali_uku/round1_train_input/youku_00000_00049_l_yuv/"+name.split('.')[0]
        new_path =os.path.join(path,name)
        save_name =out_path+".yuv"
        str ="ffmpeg -f yuv4mpegpipe -i {0} -pix_fmt yuv420p {1}".format(new_path,save_name)
        # str = "ffmpeg -f yuv4mpegpipe -i {0} -pix_fmt yuv420p {1}".format(save_name, out_path)
        os.system(str)

def yuv_y4m():
    path="E:/ali_uku/round1_train_input\youku_00000_00049_l_yuv"
    out_path="E:/ali_uku/round1_train_input\youku_00000_00049_l_toy4m"
    path_list =os.listdir(path)
    for name in path_list:
        yuv =os.path.join(path,name)
        y4m=os.path.join(out_path,name.split('.')[0]+".y4m")
        # save_name =out_path+".yuv"
        str ="ffmpeg -s 480x270 -i {0} -vsync 0 {1} -y".format(yuv,y4m)
        # str ="ffmpeg -f yuv4mpegpipe -i {0} -pix_fmt yuv420p {1}".format(new_path,save_name)
        # str = "ffmpeg -f yuv4mpegpipe -i {0} -pix_fmt yuv420p {1}".format(save_name, out_path)
        os.system(str)

if __name__=="__main__":
    yuv_y4m()