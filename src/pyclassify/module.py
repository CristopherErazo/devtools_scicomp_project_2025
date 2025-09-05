import torch
import torch.nn as nn
import torchmetrics
# import pytorch_lightning as pl
import lightning as L
from pyclassify.model import AlexNet

class Classifier(L.LightningModule):
    def __init__(self,  model: nn.Module):
        super().__init__()
        
        self.model = model
        num_classes = model.num_classes
        # self.model = AlexNet(num_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _classifier_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = self.criterion(logits, true_labels)
        preds = torch.argmax(logits, dim=1)
        return preds, true_labels, loss

    def training_step(self, batch, batch_idx):
        preds, true_labels, loss = self._classifier_step(batch)
        self.train_accuracy(preds, true_labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, true_labels, loss = self._classifier_step(batch)
        self.val_accuracy(preds, true_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        preds, true_labels, loss = self._classifier_step(batch)
        self.test_accuracy(preds, true_labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer