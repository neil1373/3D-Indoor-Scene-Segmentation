import warnings
import argparse
from lib.constants.scannet_constants import *
from lib.constants.dataset_sets import *
from lib.datasets.preprocessing.utils import *
from tqdm import tqdm

warnings.filterwarnings('ignore')
CLASS_IDs = VALID_CLASS_IDS_LONG
CLASS_LABELS = CLASS_LABELS_LONG

def get_files(mode, dataroot):
    f = open(os.path.join(dataroot, f"{mode}.txt"), "r")
    return [os.path.join(dataroot, line[:-1]) for line in f.readlines()]

def save_instance_of_stage(file_list, dataroot, mode):
    progress_bar = tqdm(file_list)
    progress_bar.set_description(f"processing tail instances in {mode}.txt")
    for fpath in progress_bar:
        df = read_plyfile(fpath)
        scene_id = os.path.basename(fpath).split('.')[0]
        instance_ids = np.unique(df[:, -1])
        for instance in instance_ids:
            segment_points = df[np.where(df[:,-1] == instance)]
            cid = int(segment_points[0][-2])
            if cid not in CLASS_IDs: continue            
            classname = CLASS_LABELS[CLASS_IDs.index(cid)]
            if classname not in TAIL_CATS_SCANNET_200:continue
            save_instance(segment_points, cid, classname, scene_id, dataroot)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='student_data')
    opt = parser.parse_args()
    return opt

def main():
    config = parse_args()
    dataroot = config.dataroot
    train_files = get_files('train', dataroot)
    # val_files = get_files('val', dataroot)

    save_instance_of_stage(train_files, dataroot, 'train')
    # save_instance_of_stage(val_files, dataroot, 'val')

if __name__ == '__main__':
    main()
    