import pickle
import os
import pandas as pd
from PIL import Image

class Reader:

    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        end = False
        observations = []
        actions = []
        pose = []
        reward = []
        done = []
        info = []
        trim = []
        episode = []

        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    observations.append(step[0])
                    actions.append(step[1])

                    md = entry['metadata']
                    pose.append(md[0])
                    reward.append(md[1])
                    done.append(md[2])
                    info.append(md[3])
                    trim.append(md[4])
                    episode.append(md[5])

            except EOFError:
                end = True

        data = [observations, actions]

        metadata = [pose, reward, done, info, trim, episode]

        return data, metadata

    def close(self):
        self._log_file.close()

path_to_home = os.environ['HOME']
path_to_proj = os.path.join(path_to_home, 'spectral-dropout')
path_dataset = os.path.join(path_to_proj, 'simulator_dataset')
path_file = os.path.join(path_dataset, 'LF_udem1_DIST_1.log')
out_dataset = os.path.join(path_to_proj, 'dataset_LanePose_sim', 'default_r1')
out_images = os.path.join(out_dataset, 'images')

f_reader = Reader(path_file)

data, metadata = f_reader.read()

for i, entry in enumerate(data[0]):
    pic_name = ''.join(['default_', str(i), '_sim.jpg'])
    pic_path = os.path.join(out_images, pic_name)
    img = Image.fromarray(entry.astype('uint8'), 'L')
    img.save(pic_path, format="JPEG")

ds = []
_, _, _, sim_info, _, _ = metadata

for i, entry in enumerate(sim_info):
    pic_name = ''.join(['default_', str(i), '_', str(i), '_sim.jpg'])
    timestamp = i
    try:
        center_distance = entry['Simulator']['lane_position']['dist']
        relative_heading = entry['Simulator']['lane_position']['angle_rad']
    except KeyError:
        center_distance = None
        relative_heading = None

    tile = None  # If we need it we can retrieve this

    ds.append([pic_name, timestamp, center_distance, relative_heading, tile])

df = pd.DataFrame(ds, columns=['ImageNumber', 'timeStamp', 'centerDistance', 'relativeHeading', 'Tile'])
path_csv = os.path.join(out_dataset, 'output_pose.csv')
df.to_csv(path_or_buf=path_csv)