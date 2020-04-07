import os
import sys
video_path = "/home/yanjianfeng/data/hive_top/download"
videos = os.listdir(video_path)
for video in videos:
    if video.find(".mp4")==-1:
        continue
    video_name,ext = os.path.splitext(video)
    full_video_path = os.path.join(video_path,video)
    img_path = os.path.join(video_path,"images")
    video_to_path = os.path.join(img_path,video_name.strip())
    if os.path.exists(video_to_path) == False:
        os.makedirs(video_to_path)
    print(video,"converting...")
    os.system("ffmpeg -i "+full_video_path+" "+video_to_path+"/thumb%04d.png")
    print(video,"converted")