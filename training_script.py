# Main execution point of the script.
if __name__ == '__main__':

    # Importing necessary libraries
    import torchvision
    import os
    import torch
    import tensorboard
    import torchvision.transforms as transforms
    from torchvision.transforms import GaussianBlur
    import numpy as np

    # Clearing GPU memory cache to avoid memory issues
    torch.cuda.empty_cache()

    # Custom dataset class for COCO Detection dataset
    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(self, img_folder, processor, train):
             # Initializing the COCO Detection dataset
            ann_file = os.path.join(img_folder, "RGB_train6.json" if train else "test.json")
            super(CocoDetection, self).__init__(img_folder, ann_file)
            self.processor = processor
            self.train = train

        def __getitem__(self, idx):
            """
            Custom getitem method to process images and targets.
            - Reads PIL image and target in COCO format.
            - Preprocesses image and target (converting target to DETR format, resizing + normalization).
            """
            img, target = super(CocoDetection, self).__getitem__(idx)
            
            # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            encoding = self.processor(images=img, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
            target = encoding["labels"][0]  # remove batch dimension

            return pixel_values, target
            
    # Loading the processor for image processing
    from transformers import DetrImageProcessor
    import pickle
    with open("/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/auxloss/resnet101/processor.pkl", "rb") as f:
        processor = pickle.load(f)

    # Preparing training and validation datasets
    train_dataset = CocoDetection(img_folder='/home/hpc/iwb3/iwb3013h/Traco/org_size/train', processor=processor, train=True)
    val_dataset = CocoDetection(img_folder='/home/hpc/iwb3/iwb3013h/Traco/org_size/val2', processor=processor, train=False)

    # DataLoader utilities for batch processing
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        """
        Custom collate function for DataLoader.
        - Pads the pixel_values and creates a batch.
        """
      pixel_values = [item[0] for item in batch]
      encoding = processor.pad(pixel_values, return_tensors="pt")
      labels = [item[1] for item in batch]
      batch = {}
      batch['pixel_values'] = encoding['pixel_values']
      batch['pixel_mask'] = encoding['pixel_mask']
      batch['labels'] = labels
      return batch

    # DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=10, shuffle=True, num_workers=32)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=10, shuffle=False, num_workers=32)

    # Defining the DETR model using PyTorch Lightning
    import pytorch_lightning as pl
    from transformers import DetrConfig, DetrForObjectDetection
    import torch

    class Detr(pl.LightningModule):
        def __init__(self, lr, lr_backbone, weight_decay):
            """
            DETR Model initialization.
            - Sets up the model parameters and DETR model.
            """
            super().__init__()
            # Initializing the DETR model
            self.model = DetrForObjectDetection.from_pretrained("/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/auxloss/resnet101/",
                                                                revision="no_timm",
                                                                num_labels=1,
                                                                ignore_mismatched_sizes=True)
            # Setting all parameters as trainable
            for param in self.model.parameters():
                param.requires_grad = True

            # Model hyperparameters
            self.lr = lr
            self.lr_backbone = lr_backbone
            self.weight_decay = weight_decay

        def forward(self, pixel_values, pixel_mask):
            """
            Forward pass of the model.
            """
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
            
    # Model instantiation
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    # Callbacks and logger setup for training
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

    # Training setup using PyTorch Lightning
    from pytorch_lightning import Trainer
    trainer = Trainer(gpus=4, max_epochs=60, gradient_clip_val=0.1, callbacks=[early_stopping_callback, checkpoint_callback], logger=logger, default_root_dir="/home/hpc/iwb3/iwb3013h/Traco/org_size/Full_trainable/five_queries/logs/")
   
    # Start training
    trainer.fit(model)



