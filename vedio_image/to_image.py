import os

# y4m to png
# y4mtobmp: ffmpeg -i xx.y4m -vsync 0 xx%3d.bmp -y
# bmptoy4m: ffmpeg -i xx%3d.bmp  -pix_fmt yuv420p  -vsync 0 xx.y4m -y
vedio_path ="E:/ali_uku/validation\youku_00150_00199_l"  #vedio path
list_vedio=os.listdir(vedio_path)
for vedio_name in list_vedio:
    str_vedio_in="E:/ali_uku/validation\youku_00150_00199_l\{0}".format(vedio_name)
    str_image_out="E:/ali_uku/validation\youku_00150_00199_l_pic\{0}".format(vedio_name.split('.')[0])
    if not os.path.exists(str_image_out):
        os.mkdir(str_image_out)
    str ="ffmpeg -i {0} -vsync 0 {1}\image%3d.bmp -y".format(str_vedio_in,str_image_out )
    os.popen(str)


# cmd = "ffmpeg -i bb_short.mp4 -vf \"select=\'eq(pict_type,  PICT_TYPE_I)\'\" -vsync vfr out%d.png"
# 'ffmpeg -i  {0} -r {1} -f image2 {2}\%05d.png'.format(video_path, frame, image_path