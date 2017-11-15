import os
from moviepy.editor import *

def extract_frames(clip, times, imgdir='tmp4/'):
    for t in times:
        imgpath = os.path.join(imgdir, '{}.png'.format(t))
        clip.save_frame(imgpath, t=t/50)



videofilein = 'project_video.mp4'

clip1 = VideoFileClip(videofilein)#.subclip(5,25)

extract_frames(clip1, range(42*50,51*50))