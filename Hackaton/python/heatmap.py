import numpy as np
from PIL import Image

def read_int(f):
    ba = bytearray(4)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.int32)
    return prm[0]

def read_double(f):
    ba = bytearray(8)
    f.readinto(ba)
    prm = np.frombuffer(ba, dtype=np.double)
    return prm[0]

def read_double_tab(f, n):
    ba = bytearray(8*n)
    nr = f.readinto(ba)
    if nr != len(ba):
        return []
    else:
        prm = np.frombuffer(ba, dtype=np.double)
        return prm

def get_pics_from_file(filename):
    with open(filename, "rb") as f:
        # Get info header
        info = {}
        info["nb_pics"] = read_int(f)
        info["freq_sampling_khz"] = read_double(f)
        info["freq_trame_hz"] = read_double(f)
        info["freq_pic_khz"] = read_double(f)
        info["norm_fact"] = read_double(f)

        # Parse pics
        pics = []
        while True:
            item = read_double_tab(f, info["nb_pics"])
            if len(item) != info["nb_pics"]:
                break
            item = np.array(item)
            pics.append(np.array(item))
        pics = np.stack(pics, axis=0)
        return pics, info

if __name__ == '__main__':
    pics_pad0, info = get_pics_from_file("../data/pics_NOKEY.bin")
    pics_a, info = get_pics_from_file("../data/pics_CTRL.bin")

    img = Image.new('L', (info["nb_pics"], len(pics_pad0)))
    pixels = []

    for i, frame in enumerate(pics_a):
        noise_frame = pics_pad0[i]

        for i, p in enumerate(frame):
            result = p - noise_frame[i]

            if result < 0:
                result = 0

            pixels.append(int(result * 255))

    img.putdata(pixels)
    img.save("file.png")
