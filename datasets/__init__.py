from .kitti import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .rsrd import RSRDDataset
__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "rsrd": RSRDDataset
}
