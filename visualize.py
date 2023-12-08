import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch
import matplotlib.pyplot as plt
import torchvision
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Custom DETR model class inheriting from PyTorch LightningModule
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        """
        Initializes the DETR model with custom configuration.
        Args:
            lr: Learning rate for optimizer.
            lr_backbone: Learning rate for the backbone network.
            weight_decay: Weight decay for regularization in the optimizer.
        """
        super().__init__()
        # Initialization of DETR model with custom parameters
        self.model = DetrForObjectDetection.from_pretrained(
            "C:/Salil Data/Salil/Salil FAU/MSc ACES/Sem 2 Courses/Tracking Olympiad/Trained/DETR resnet50 default/Full_trainable/5_queries/auxloss_true",
            revision="no_timm",  # Specifies not to use the timm library
            num_labels=1,        # Number of labels in the dataset
            ignore_mismatched_sizes=True)
        # Making all parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True

        # Model hyperparameters
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

# Instantiate the model with specified hyperparameters
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

# Loading a saved model checkpoint
checkpoint_path = "C:/Salil Data/Salil/Salil FAU/MSc ACES/Sem 2 Courses/Tracking Olympiad/Trained/DETR resnet50 default/Full_trainable/5_queries/auxloss_true/bboxloss_inc/logstensorboard/DETR_Full_trainable_ogr_size_5_queries/version_2/checkpoints/epoch=24-step=3700.ckpt"
checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))

# Load the model state dictionary from the checkpoint
model.load_state_dict(checkpoint['state_dict'])

# Custom dataset class for COCO Detection
class CocoDetection(torchvision.datasets.CocoDetection):
    # Constructor with image folder and processor parameters
    # Other method definitions like __getitem__ for dataset processing
    def __init__(self, img_folder, processor, purpose=None):
        if purpose == 'train':
            ann_file = os.path.join(img_folder, "RGB_train.json")
        if purpose == 'val':
            ann_file = os.path.join(img_folder, "RGB_val.json")
        if purpose == 'test':
            ann_file = os.path.join(img_folder, "test.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

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

# Initialize the image processor
from transformers import DetrImageProcessor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Create instances of the dataset for training, validation, and testing
train_dataset = CocoDetection(img_folder='C:\\Salil Data\\Salil\\Salil FAU\\MSc ACES\\Sem 2 Courses\\Tracking Olympiad\\Train data\\Training data\\org_size\\new_RGB_data1\\train', processor=processor, purpose='train')
val_dataset = CocoDetection(img_folder='C:\\Salil Data\\Salil\\Salil FAU\\MSc ACES\\Sem 2 Courses\\Tracking Olympiad\\Train data\\Training data\\org_size\\new_RGB_data1\\val', processor=processor, purpose='val')
test_dataset = CocoDetection(img_folder='C:\\Salil Data\\Salil\\Salil FAU\\MSc ACES\\Sem 2 Courses\\Tracking Olympiad\\Train data\\Training data\\org_size\\new_RGB_data1\\test_for_train5', processor=processor, purpose='test')

# Custom collate function for DataLoader
def collate_fn(batch):
    # Function to collate data batches for DataLoader
    # Returns a dictionary with processed batch data
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

# DataLoaders for training, validation, and testing datasets
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=1)

import matplotlib.pyplot as plt
from PIL import Image

# Function to plot and save results of model predictions
def plot_results(pil_img, scores, labels, boxes, index):
    # Function details with explanations on the plotting process
    print(pil_img.size)
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        #         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                                    fill=False, color=c, linewidth=3))
        center = (xmin + 32.5, ymin + 32.5)
        print(center)
        radius = 15
        ax.add_artist(plt.Circle(center, radius, edgecolor='black', facecolor='blue'))
        #         text = f'{model.config.id2label[label]}: {score:0.2f}'
        text = f'Hexbug head: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=7,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("C:\\Salil Data\\Salil\\Salil FAU\\MSc ACES\\Sem 2 Courses\\Tracking Olympiad\\Trained\\DETR resnet50 default\\Full_trainable\\5_queries\\auxloss_true\\bboxloss_inc\\Gaussian_Color\\valProbs_V27\\image_" + str(index) + ".jpg")
    plt.show()

index = len(test_dataset)

# Iterating over the test dataset to get predictions and plot results
for ind in range(index):
    # Process and get predictions for each image in the test dataset
    # Use the plot_results function to visualize and save the results
    pixel_values, target = test_dataset[ind]
    print(target['boxes'])
    device = torch.device( "cuda")
    pixel_values = pixel_values.unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
      # forward pass to get class logits and bounding boxes
      outputs = model(pixel_values=pixel_values, pixel_mask=None)
    # print("Outputs:", outputs.keys())

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    # load image based on ID
    image_id = target['image_id'].item()
    image = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join('C:\\Salil Data\\Salil\\Salil FAU\\MSc ACES\\Sem 2 Courses\\Tracking Olympiad\\Train data\\Training data\\org_size\\new_RGB_data1\\test_for_train5\\', image['file_name']))
    # postprocess model outputs
    width, height = image.size
    postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                    target_sizes=[(height, width)],
                                                                    threshold=0.5)
    results = postprocessed_outputs[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'], ind)
