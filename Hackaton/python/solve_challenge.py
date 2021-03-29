import numpy as np
import glob
import random
from PIL import Image
from read_pics import get_pics_from_file

chunks = [
    (1481, 1836),
    (1837, 2140),
    (3011, 3222),
    (3223, 3323),
    (3324, 3677),
    (3679, 3781),
    (3786, 4032),
    (4033, 4141),
    (4143, 4421),
    (4423, 4563),
    (4565, 4821),
    (4824, 4970),
    (4973, 5067),
    (5909, 6058),
    (6272, 6419),
    (6617, 6735),
    (7024, 7155),
    (7425, 7539),
    (7813, 7931),
    (8178, 8297),
    (8552, 8670),
    (8897, 9028),
    (9188, 9336),
    (9539, 9670),
    (9839, 9982),
]

class Key:
    def __init__(self, name, mean):
        self.name = name
        self.mean = mean

    def __str__(self):
        return f"Key(name='{self.name}', mean={self.mean})"

    def __repr__(self):
        return self.__str__()


def mean_slice(s):
    acc = None

    for e in s:
        if acc is None:
            acc = e
        else:
            acc += e

    return acc / len(s)


class GenFuzz:
    MAX_KEY_COUNT = 2

    def __init__(self, corpus, target, population=100):
        self.corpus = corpus
        self.target = target
        self.entities = []
        self.population = population

    def __generate_random_pop(self):
        pass

if __name__ == '__main__':
    pics_nokey, info = get_pics_from_file("../data/pics_NOKEY.bin")
    keys = []

    for key_file in glob.glob("../data/pics_*.bin"):
        if "LOGINMDP" in key_file or "NOKEY" in key_file:
            continue

        key = key_file.split("pics_")[1].replace(".bin", "")
        pics_cur, info = get_pics_from_file(key_file)
        acc = None

        for i, cur_frame in enumerate(pics_cur):
            if i >= len(pics_nokey):
                break

            nokey_frame = pics_nokey[i]
            denoised = cur_frame - nokey_frame

            if acc is None:
                acc = denoised
            else:
                acc += denoised

        acc /= len(pics_cur)
        keys.append(Key(key, acc))

    pics_login, info = get_pics_from_file("../data/pics_LOGINMDP.bin")

    # Denoise the data
    for i, cur_frame in enumerate(pics_login):
        pics_login[i] = cur_frame - pics_nokey[i]

    # Test first slice
    for low, high in chunks:
        first_slice = pics_login[low:high]
        first_slice_mean = mean_slice(first_slice)

        # Try to find the best candidate combination
        nokey_mean = np.zeros(len(first_slice_mean), dtype=np.double)

        right_keys = keys + [Key("NOKEY", nokey_mean)]
        min_score = 1000000.0

        best = None

        # Key one
        for i in range(len(keys)):
            for y in range(len(right_keys)):
                sum_of_means = (keys[i].mean + right_keys[y].mean)
                score = np.linalg.norm(first_slice_mean - sum_of_means)

                if score < min_score:
                    min_score = score
                    best = [keys[i], right_keys[y]]

        result = list(filter(lambda a: a.name not in ["NOKEY"], best))

        for key in result:
            print(key.name, end=" ")

        print("")
