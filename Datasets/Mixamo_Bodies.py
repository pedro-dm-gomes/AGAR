import os
import numpy as np


class Bodys(object):
    def __init__(
        self,
        root="/home/pedro/Desktop/Datasets/NPYs",
        seq_length=12,
        num_points=4000,
        train=True,
    ):


        letters = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]  # 25
        always_letters = [
            "B",
            "C",
            "D",
            "F",
            "G",
            "H",
            "I",
            "J",
            "P",
            "R",
            "T",
            "W",
        ]  # 12
        persons = [
            "Megan",
            "Sophie",
            "Brian",
            "Douglas",
            "Joe",
            "Kate",
            "Lewis",
            "Astra",
            "Louise",
            "Malcom",
            "Martha",
            "Remmy",
            "Regina",
            "Roth",
            "Stefani",
        ]  # 15 sequencias

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []
        self.data_color = []

        print("seq_length", seq_length)
        print("num_points", num_points)
        folder = str(num_points)

        log_nr = 0
        print("-- Mixamo Dataset ")

        if train:
            splits = [folder]
        else:
            splits = ["test"]

        for split in splits:
            print("split: ", split)
            split_path = os.path.join(root, split)
            split_path_color = os.path.join(root_color, split)

            # SELECT CHARACTER
            for charater in sorted(os.listdir(split_path)):
                charater_path = os.path.join(split_path, charater)
                charater_path_color = os.path.join(split_path_color, charater)

                if charater != "zBrian":
                    rnd_0 = np.random.randint(0, 12)
                    rnd_01 = np.random.randint(0, 12)
                    rnd_2 = np.random.randint(0, 25)
                    rnd_1 = np.random.randint(0, 25)
                    rnd_3 = np.random.randint(0, 25)

                    print(
                        "[ ",
                        charater,
                        " ] Load: ",
                        letters[rnd_1],
                        letters[rnd_2],
                        always_letters[rnd_0],
                        always_letters[rnd_01],
                    )
                    for sequence in sorted(os.listdir(charater_path)):
                        if sequence[0] != "0":  # Load all data
                            fps = np.random.randint(1, 4)
                            odd = np.random.randint(0, 2)
                            sequence_path = os.path.join(charater_path, sequence)
                            print("[%10d] [%s] (1/%d fps)" % (log_nr, sequence, fps))

                            # LOAD POINTS
                            log_data = []
                            frame = odd
                            for npy in sorted(os.listdir(sequence_path)):
                                # Load at diferent speeds
                                if frame % (fps) == 0:
                                    npy_file = os.path.join(sequence_path, npy)
                                    npy_data = np.load(npy_file)
                                    log_data.append(npy_data)
                                frame = frame + 1
                            self.data.append(log_data)

                            log_nr = log_nr + 1

        print("self.data", np.shape(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):
        nr_seq = len(self.data)
        rand = np.random.randint(0, nr_seq)
        log_data = self.data[rand]

        total_lenght = len(log_data)
        start_limit = total_lenght - self.seq_length

        # CHECK FOR ERROR
        if start_limit < 0 or start_limit > 40:
            error = 1
            # get new sequence
            while error == 1:
                rand = np.random.randint(0, nr_seq)
                log_data = self.data[rand]
                total_lenght = len(log_data)
                start_limit = total_lenght - self.seq_length
                if start_limit > -1 and start_limit < 40:
                    error = 0
                else:
                    error = 1

        if start_limit == 0:
            start = 0
        else:
            start = np.random.randint(0, start_limit)

        cloud_sequence = []

        for i in range(start, start + self.seq_length):
            pc = log_data[i]
            npoints = pc.shape[0]
            cloud_sequence.append(pc)

        points = np.stack(cloud_sequence, axis=0)

        return points
