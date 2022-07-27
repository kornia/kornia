import hydra
import torch
import torchvision
import torchvision.transforms as T
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

import kornia as K
from kornia.x import Configuration, ModelCheckpoint, ObjectDetectionTrainer

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Configuration)


@hydra.main(config_path=".", config_name="config.yaml")
def my_app(config: Configuration) -> None:

    # create the model
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    def collate_fn(data):
        # To map from [{A, B}, ...] => [{A}, ...], [{B}, ...]
        return [d[0] for d in data], [d[1] for d in data]

    # create the dataset
    train_dataset = torchvision.datasets.WIDERFace(
        root=to_absolute_path(config.data_path), transform=T.ToTensor(), split='train', download=False
    )

    valid_dataset = torchvision.datasets.WIDERFace(
        root=to_absolute_path(config.data_path), transform=T.ToTensor(), split='val', download=False
    )

    # create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    valid_daloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    # create the loss function
    criterion = None

    # instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs * len(train_dataloader))

    # define some augmentations
    _augmentations = K.augmentation.AugmentationSequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomAffine(
            degrees=0.0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),  # NOTE: XYXY bbox format cannot handle rotated boxes
        data_keys=['input', 'bbox_xyxy'],
    )

    def bbox_xywh_to_xyxy(boxes: torch.Tensor):
        boxes[..., 2] = boxes[..., 0] + boxes[..., 2]  # x + w
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]  # y + h
        return boxes

    def preprocess(self, x: dict) -> dict:
        x['target'] = {
            "boxes": [bbox_xywh_to_xyxy(a['bbox'].float()) for a in x['target']],
            # labels are set to 1 for all faces
            "labels": [torch.tensor([1] * len(a['bbox'])) for a in x['target']],
        }
        return x

    def augmentations(self, sample: dict) -> dict:
        xs, ys, ys2 = [], [], []
        for inp, trg, lab in zip(sample["input"], sample["target"]["boxes"], sample["target"]["labels"]):
            x, y = _augmentations(inp[None], trg[None])
            xs.append(x[0])
            ys.append(y[0])
            ys2.append(lab)
        return {"input": xs, "target": {"boxes": ys, "labels": ys2}}

    def on_before_model(self, sample: dict) -> dict:
        return {
            "input": sample["input"],
            "target": [
                {"boxes": v, "labels": l} for v, l in zip(sample["target"]["boxes"], sample["target"]["labels"])
            ],
        }

    model_checkpoint = ModelCheckpoint(filepath="./outputs", monitor="map")

    trainer = ObjectDetectionTrainer(
        model,
        train_dataloader,
        valid_daloader,
        criterion,
        optimizer,
        scheduler,
        config,
        num_classes=81,
        loss_computed_by_model=True,
        callbacks={
            "preprocess": preprocess,
            "augmentations": augmentations,
            "on_before_model": on_before_model,
            "on_checkpoint": model_checkpoint,
        },
    )
    trainer.fit()
    trainer.evaluate()


if __name__ == "__main__":
    my_app()
