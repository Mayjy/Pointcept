import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset

@DATASETS.register_module()
class aQcKITTIDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        split2seq = dict(
            train=[0],  
            val=[1],  
            test=[2],  
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 3)  # 只读取 XYZ
        coord = scan[:, :3]  # 只保留 XYZ

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int32)
        else:
            segment = np.full(scan.shape[0], self.ignore_index, dtype=np.int32)  # 其他类别填充 ignore_index

        data_dict = dict(
            coord=coord,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        """类别映射"""
        return {
            0: ignore_index,  # 其他类别（0）忽略
            1: 0,  # spreader → 0
            2: 1,  # cell_guide → 1
        }

    @staticmethod
    def get_learning_map_inv(ignore_index):
        """反向类别映射"""
        return {
            ignore_index: ignore_index,  # 其他类别仍然忽略
            0: 1,  # 反向映射 0 → spreader (1)
            1: 2,  # 反向映射 1 → cell_guide (2)
        }
