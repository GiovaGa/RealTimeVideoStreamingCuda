ffplay -f video4linux2 -video_size 1280x720 -i /dev/video2
# equivalente: ffplay -f video4linux2 /dev/video0

# To see available formats for devices:
ffmpeg -f v4l2 -list_formats all -i /dev/video0

# To see al available devices:
v4l2-ctl --list-devices


# get raw data and convert
ffmpeg -f rawvideo -video_size 1920x1080 -pix_fmt yuyv422 -i out out.jpg

# Everything in a single line:
# gcc capture0.c -Wall && ./a.out -f -d /dev/video0 -o | ffmpeg -f rawvideo -video_size 1920x1080 -pix_fmt yuyv422 -i /dev/stdin out.jpg


# send to network on port 48550
ffmpeg -i /dev/video0 -framerate 15 -vcodec libx264 -preset ultrafast -tune zerolatency -maxrate 750k -bufsize 3000k -f mpegts udp://localhost:48550
# receive from network port 48550
ffplay -i "udp://localhost:48550"
