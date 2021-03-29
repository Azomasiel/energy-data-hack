import numpy as np
from read_pics import get_pics_from_file
from PIL import Image

if __name__ == '__main__':
    pics_pad0, info = get_pics_from_file("../data/pics_NOKEY.bin")
    pics_a, info = get_pics_from_file("../data/pics_LOGINMDP.bin")

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
