import numpy as np
import glob
import random
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


def solve():
    login, _ = get_pics_from_file("./data/pics_LOGINMDP.bin")
    noise, _ = get_pics_from_file("./data/pics_NOKEY.bin")

    mean_noise = np.mean(noise, axis=0)
    login_denoised = np.subtract(login, mean_noise)

    keys = []

    # Load all key data
    for key_file in glob.glob("./data/pics_*.bin"):
        if "LOGINMDP" in key_file or "NOKEY" in key_file:
            continue

        key = key_file.split("pics_")[1].replace(".bin", "")
        pics_cur, info = get_pics_from_file(key_file)

        key_mean_denoise = np.mean(np.subtract(pics_cur, mean_noise), axis=0)
        keys.append(Key(key, key_mean_denoise))

    second_keys = keys + [Key("NOKEY", np.zeros(len(mean_noise), dtype=np.double))]

    # Associate key(s) to a chunk
    for low, high in chunks:
        chunk_mean = np.mean(login_denoised[low:high], axis=0)
        min_score = 100000.0
        best_keys = []

        for first_key in keys:
            for second_key in second_keys:
                score = np.linalg.norm(chunk_mean - (first_key.mean + second_key.mean))

                if score < min_score:
                    min_score = score
                    best_keys.append((score, [first_key, second_key]))

        best_keys = sorted(best_keys, key=lambda a: a[0])

        print(f"chunks[{low},{high}]:")

        for i in range(3):
            candidate = best_keys[i]
            print(f"  Candidate {i+1} (score: {candidate[0]}):", end="")

            for key in candidate[1]:
                if key.name not in ["NOKEY"]:
                    print(key.name, end=" ")

            print("")


if __name__ == '__main__':
    solve()
