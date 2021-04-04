import numpy as np
import glob
import random
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage
from read_pics import get_pics_from_file


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
    login_denoised_smooth = ndimage.gaussian_filter1d(login_denoised, 10, 0)

    # Split the input into segments
    ac = AgglomerativeClustering(compute_full_tree=True, distance_threshold=9, n_clusters=None)
    yac = ac.fit_predict(login_denoised_smooth)

    print(f"Cluster count: {ac.n_clusters_}")

    chunks_index = [0]
    for i in range(yac.size - 1):
        if yac[i] != yac[i+1]:
            chunks_index.append(i)

    chunks_index.append(yac.size)
    chunks = [(x,y) for x,y in zip(chunks_index[:-1], chunks_index[1:])]

    print(len(chunks))

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
    selected_keys = [] # List of the selected best keys

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

            presses = []

            for key in candidate[1]:
                if key.name not in ["NOKEY"]:
                    print(key.name, end=" ")
                    presses.append(key.name)

            if i == 0:
                selected_keys.append("+".join(presses))

            print("")

    print(f"Selected keys: {', '.join(selected_keys)}")


if __name__ == '__main__':
    solve()
