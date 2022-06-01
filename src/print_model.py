import torch
from summary import summary



feature_extractor = torch.load('runs/classifier.pt', map_location='cpu')

summary(feature_extractor, 48)