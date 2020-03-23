from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class FCOS(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(FCOS, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)
        for pred in preds:
            for p in pred:
                condi = p.max() < 1e8 and p.min() > -1e8
                print("max: {}, min: {}, condition: {}".format(p.max(), p.min(), condi))
                # assert p.max() < 1e8 and p.min() > -1e8
                # condi_1 = p.max() < 1e8
                # condi_2 = p.min() > -1e8
                # condi_3 = condi_1 and condi_2

                if not condi:
                    tmp = p.detach().cpu().numpy()
                    p_max = 1e8
                    p_min = -1e8
                    p_max = p.max() < p_max
                    p_min = p.min() > p_min
                    print(tmp.shape)

        if return_loss:
            preds_gt = preds + (example["gt_boxes"],
                                example["gt_classes"])
            return self.bbox_head.loss(*preds_gt)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
