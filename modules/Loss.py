import torch.nn as nn

def model_loss(outputs, labels, l1, l2):
    criterion = nn.CrossEntropyLoss()
    class_loss = criterion(outputs, labels)
    loss = class_loss + 0.3 * l1 + 0.3 * l2
    return loss