from moviepy.editor import *

VideoFileClip('mov.mp4').speedx(4).write_gif('out.gif') # 4배속도로 gif 생성

