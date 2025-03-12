from linuxpy.video.device import Device

with Device.from_id(0) as cam:
    for i, frame in enumerate(cam):
        print(f"frame #{i}: {len(frame)} bytes\t", type(frame))
        if i > 9:
            break

