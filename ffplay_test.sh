ffplay -f video4linux2 -video_size 1280x720 -i /dev/video0
# equivalente: ffplay -f video4linux2 /dev/video0

# To see available formats for devices:
# ffmpeg -f v4l2 -list_formats all -i /dev/video0

# To see al available devices:
# v4l2-ctl --list-devices
