import rospy
import rosbag
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import torch
from mmcv import Config
from pointcept.models import build_model
import std_msgs.msg

def load_model(config_path, weight_path):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model = model.cuda().eval()
    return model, cfg

def label_to_color(labels):
    # 简单的colormap（可自定义）
    cmap = np.array([
        [255, 0, 0],    # 0: red
        [0, 255, 0],    # 1: green
        [0, 0, 255],    # 2: blue
        [255, 255, 0],  # 3: yellow
        [0, 255, 255],  # 4: cyan
        [255, 0, 255],  # 5: magenta
        [255, 255, 255] # 6: white
    ], dtype=np.uint8)
    return cmap[labels % len(cmap)]

def infer_one_frame(model, points):
    input_dict = {
        "coord": torch.from_numpy(points[:, :3]).float().unsqueeze(0).cuda(),
        "strength": torch.from_numpy(points[:, 3]).float().unsqueeze(0).cuda(),
    }
    with torch.no_grad():
        output = model(input_dict)
        pred = output["seg_logits"].max(-1)[1].cpu().numpy()[0]  # (N,)
    return pred

def publish_semantic_pointcloud(pub, header, points, labels):
    colors = label_to_color(labels)
    points_with_color = np.hstack([points[:, :3], colors])
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.UINT8, 1),
        PointField('g', 13, PointField.UINT8, 1),
        PointField('b', 14, PointField.UINT8, 1),
    ]
    pc2_msg = pc2.create_cloud(header, fields, points_with_color)
    pub.publish(pc2_msg)

if __name__ == "__main__":
    # 1. 配置文件和权重路径
    config_path = "configs/aqc/semseg-pt-v3m1-4-train.py"
    weight_path = "exp/aqc/semseg-pt-v3m1-4-train/model/model_best.pth"
    # 2. 加载模型
    model, cfg = load_model(config_path, weight_path)

    # 3. 初始化ROS节点和发布器
    rospy.init_node('semantic_inference')
    pub = rospy.Publisher('/semantic_points', PointCloud2, queue_size=1)

    # 4. 读取rosbag并推理
    bag = rosbag.Bag('your.bag')
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        points = np.array(list(pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)))
        if points.shape[0] == 0:
            continue
        # 补一列强度
        strength = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack([points, strength])  # (N, 4)
        publish_semantic_pointcloud(pub, msg.header, points, labels)
        rospy.sleep(0.1)  # 控制发布频率

    bag.close()