
import os


in_path ="E:/ali_uku/round1_train_result/0.y4m"
in_path2 ="E:/ali_uku/validation\youku_00150_00199_h_GT/Youku_00150_h_GT.y4m"
str ="ffmpeg -i {0}  -i {1}  -lavfi psnr=\"stats_file=psnr.log\" -f null -".format(in_path, in_path2)      #
# str ="ffmpeg -i {0} {1}\image%d.png".format(str_vedio_in, str_image_out)
os.system(str)