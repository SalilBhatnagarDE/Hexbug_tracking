if __name__ == '__main__':

    import torchvision
    import os
    import torch
    import tensorboard
    import torchvision.transforms as transforms
    from torchvision.transforms import GaussianBlur
    import numpy as np

    torch.cuda.empty_cache()

    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(self, img_folder, processor, train):
            ann_file = os.path.join(img_folder, "RGB_train6.json" if train else "test.json")
            super(CocoDetection, self).__init__(img_folder, ann_file)
            self.processor = processor
            self.train = train

        def __getitem__(self, idx):
            # read in PIL image and target in COCO format
            # feel free to add data augmentation here before passing them to the next step
            img, target = super(CocoDetection, self).__getitem__(idx)

            # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            encoding = self.processor(images=img, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
            target = encoding["labels"][0]  # remove batch dimension

            return pixel_values, target

    from transformers import DetrImageProcessor

    import pickle
    with open("/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/auxloss/resnet101/processor.pkl", "rb") as f:
        processor = pickle.load(f)

    train_dataset = CocoDetection(img_folder='/home/hpc/iwb3/iwb3013h/Traco/org_size/train', processor=processor, train=True)
    val_dataset = CocoDetection(img_folder='/home/hpc/iwb3/iwb3013h/Traco/org_size/val2', processor=processor, train=False)


    from torch.utils.data import DataLoader

    def collate_fn(batch):
      pixel_values = [item[0] for item in batch]
      encoding = processor.pad(pixel_values, return_tensors="pt")
      labels = [item[1] for item in batch]
      batch = {}
      batch['pixel_values'] = encoding['pixel_values']
      batch['pixel_mask'] = encoding['pixel_mask']
      batch['labels'] = labels
      return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=10, shuffle=True, num_workers=32)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=10, shuffle=False, num_workers=32)

    import pytorch_lightning as pl
    from transformers import DetrConfig, DetrForObjectDetection
    import torch

    class Detr(pl.LightningModule):
        def __init__(self, lr, lr_backbone, weight_decay):
            super().__init__()
            # replace COCO classification head with custom head
            # we specify the "no_timm" variant here to not rely on the timm library
            # for the convolutional backbone
            self.model = DetrForObjectDetection.from_pretrained("/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/auxloss/resnet101/",
                                                                revision="no_timm",
                                                                num_labels=1,
                                                                ignore_mismatched_sizes=True)
            # Set all parameters as trainable
            for param in self.model.parameters():
                param.requires_grad = True

            # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
            self.lr = lr
            self.lr_backbone = lr_backbone
            self.weight_decay = weight_decay

        def forward(self, pixel_values, pixel_mask):
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            return outputs

        def common_step(self, batch, batch_idx):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

            loss = outputs.loss
            loss_dict = outputs.loss_dict

            return loss, loss_dict

        def training_step(self, batch, batch_idx):
            loss, loss_dict = self.common_step(batch, batch_idx)
            # logs metrics for each training_step,
            # and the average across the epoch
            self.log("training_loss", loss)
            for k, v in loss_dict.items():
                self.log("train_" + k, v.item())

            return loss

        def validation_step(self, batch, batch_idx):
            loss, loss_dict = self.common_step(batch, batch_idx)
            self.log("validation_loss", loss)
            for k, v in loss_dict.items():
                self.log("validation_" + k, v.item())

            return loss

        def configure_optimizers(self):
            param_dicts = [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                          weight_decay=self.weight_decay)

            return optimizer

        def train_dataloader(self):
            return train_dataloader

        def val_dataloader(self):
            return val_dataloader

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    early_stopping_callback = EarlyStopping(monitor='validation_loss', patience=15, mode='min')

    checkpoint_callback = ModelCheckpoint(
    dirpath="/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/logstensorboard/DETR_Full_trainable_ogr_size_5_queries/",
    filename="detr-{epoch:02d}-{validation_loss:.4f}",
    monitor='validation_loss',
    mode='min',
    save_top_k=1,
    save_last=True)

    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/logstensorboard/", name="DETR_Full_trainable_ogr_size_5_queries")

    from pytorch_lightning import Trainer

    trainer = Trainer(gpus=4, max_epochs=60, gradient_clip_val=0.1, callbacks=[early_stopping_callback, checkpoint_callback], logger=logger, default_root_dir="/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/logs/")

    trainer.fit(model)



