import moviepy.editor as mp
import sys
import shutil
import os

if os.path.isdir('audios-generados'):
    shutil.rmtree('audios-generados')

os.mkdir('audios-generados')
file_name = sys.argv[1]
time_one = 0
time_two = 5
count = 0
video = mp.VideoFileClip(file_name)
duration = video.duration
while time_one < duration:
    if (time_two < duration):
        clip = mp.VideoFileClip(file_name).subclip(time_one,time_two)
        clip.audio.write_audiofile("audios-generados/audio-%d.wav"%count)
        time_one = time_two
        time_two = time_two + 5
        count = count + 1
    else:
        time_two = duration
        clip = mp.VideoFileClip(file_name).subclip(time_one,time_two)
        clip.audio.write_audiofile("audios-generados/audio-%d.wav"%count)
        time_one = time_two
        time_two = time_two + 5
        count = count + 1