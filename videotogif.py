from moviepy.editor import *

file_nm = sys.argv[1]

VideoFileClip(file_nm).speedx(6).write_gif(file_nm + '.gif')

