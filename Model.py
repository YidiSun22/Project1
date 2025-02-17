import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

# ---- MLP Block ----
class MLPBlock(nn.Module):
    def __init__(self, in_features=256, hidden_dim=2048, dropout=0.5):
        super(MLPBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_features)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, hidden_dim)  # 2048 MLP

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.global_max_pool(x)  # Shape: (B, C, 1, 1)
        x = torch.flatten(x, start_dim=1)  # Flatten to (B, C)
        x = self.dropout(x)
        x = self.fc(x)
        return x




# ---- FC Network (Final Classifier) ----
class FCNetwork(nn.Module):
    def __init__(self, input_dim=4 * 2048, hidden_dim1=2048, hidden_dim2=1024, num_classes=15):
        super(FCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 2048 MLP
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 1024 MLP
        self.fc3 = nn.Linear(hidden_dim2, num_classes)  # num_classes
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # BN layer (2048)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)  # BN Layer (1024)
        self.sigmoid = nn.Sigmoid()  # num_classes probabilities 

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  
        return self.sigmoid(x)  

# ---- CheXNet Backbone (DenseNet121) ----
class CheXNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(CheXNetBackbone, self).__init__()
        
        # loading pre-trained model
        densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = densenet.features  # extract DenseNet feature map
        
    def forward(self, x):
        x = self.features[0](x)    # Conv1
        x = self.features[1](x)    # BatchNorm
        x = self.features[2](x)    # ReLU
        x = self.features[3](x)    # MaxPool

        c2 = self.features[4](x)   # Dense Block 1
        x = self.features[5](c2)   # Transition 1

        c3 = self.features[6](x)   # Dense Block 2
        x = self.features[7](c3)   # Transition 2

        c4 = self.features[8](x)  # Dense Block 3
        x = self.features[9](c4)  # Transition 3

        c5 = self.features[10](x)  # Dense Block 4
        
        return c2, c3, c4, c5


import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 1024], feature_size=256):
        super(FPN, self).__init__()

        # **1x1 pointwise convolution to reduce dimensions of DenseNet features**
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, feature_size, kernel_size=1) for in_ch in in_channels_list
        ])

        # **2x2 Transposed Convolution for Upsampling (following the architecture diagram)**
        self.upsample_convs = nn.ModuleList([
            nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2) for _ in range(3)
        ])

        # **3x3 Convolution to smooth FPN outputs**
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1) for _ in range(4)
        ])

    def forward(self, features):
        """Input features = (p2, p3, p4, p5), returns FPN-processed feature maps"""
        c2, c3, c4, c5 = features

        # **1x1 Pointwise Convolution for Dimensionality Reduction**
        p2, p3, p4, p5 = [conv(f) for conv, f in zip(self.lateral_convs, [c2, c3, c4, c5])]

        # **Top-down feature fusion in FPN**
        p4 = p4 + self.upsample_convs[0](p5)  # `P5 → P4`
        p3 = p3 + self.upsample_convs[1](p4)  # `P4 → P3`
        p2 = p2 + self.upsample_convs[2](p3)  # `P3 → P2`

        # **Final Outputs**
        p2 = self.smooth_convs[0](p2)
        p3 = self.smooth_convs[1](p3)
        p4 = self.smooth_convs[2](p4)
        p5 = self.smooth_convs[3](p5)
        
        return p2, p3, p4, p5


class CheXNetFPNModel(nn.Module):
    def __init__(self, num_classes=15):
        super(CheXNetFPNModel, self).__init__()
        self.backbone = CheXNetBackbone(pretrained=True)
        self.fpn = FPN()

        # **Add `p6` computation using 2x2, stride=2 convolution**
        self.conv_p6 = nn.Conv2d(1024, 256, kernel_size=2, stride=2)
        self.conv_p7 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # MLP processing for P2, P3, P4, P5
        self.mlp_p2 = MLPBlock(in_features=256)
        self.mlp_p3 = MLPBlock(in_features=256)
        self.mlp_p4 = MLPBlock(in_features=256)
        self.mlp_p5 = MLPBlock(in_features=256)
        
        self.mlp_c5 = MLPBlock(in_features=1024)
        self.mlp_p6 = MLPBlock(in_features=256)
        self.mlp_p7 = MLPBlock(in_features=256)
            
        # **Batch Normalization**
        self.bn = nn.BatchNorm1d(7 * 2048)  # BN layer to process concatenated features

        # **Fully Connected Network for final classification**
        self.classifier = FCNetwork(input_dim=7 * 2048, num_classes=num_classes)

    def forward(self, x):
        # **Step 1: Extract DenseNet features**
        c2, c3, c4, c5 = self.backbone(x)

        # **Step 2: Compute FPN features**
        p2, p3, p4, p5 = self.fpn((c2, c3, c4, c5))

        # **Step 3: Process c5, upper structure part**
        p6 = self.conv_p6(c5)
        p7 = self.relu(self.conv_p7(p6))

        # **Step 4: Extract features using MLP**
        p2_feat = self.mlp_p2(p2)  # `[batch, hidden_dim]`
        p3_feat = self.mlp_p3(p3)
        p4_feat = self.mlp_p4(p4)
        p5_feat = self.mlp_p5(p5)
    
        c5_feat = self.mlp_c5(c5)
        p6_feat = self.mlp_p6(p6)
        p7_feat = self.mlp_p7(p7)
        
        # **Step 5: Concatenate all features**
        concatenated_features = torch.cat([p2_feat, p3_feat, p4_feat, p5_feat, c5_feat, p6_feat, p7_feat], dim=1)  # `[batch, 7×hidden_dim]`

        # **Step 6: Batch Normalization**
        normalized_features = self.bn(concatenated_features)

        # **Step 7: Classification using Fully Connected Network**
        output = self.classifier(normalized_features)  # **Final `Sigmoid` output with `num_classes` dimensions**

        return output
