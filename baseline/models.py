import torch
import torch.nn as nn
import torchvision.models as models

class DetectEdgeNN(nn.Module):
    """
       Model that predicts whether there exists a relationship between
       a subject and an object.
    """

    def __init__(self):
        """
           Loads the pretrained DenseNet-121 model and replaces the last
           fully connected layer.
        """
        super(DetectEdgeNN, self).__init__()
        densenet = models.densenet121(pretrained=True)
        in_features = densenet.classifier.in_features
        modules = list(densenet.children())[:-1] # delete the last fc layer.
        modules.append(nn.AvgPool2d(7)) # add back the average pooling
        self.cnn = nn.Sequential(*modules)
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(3*in_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
        """
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, sub_images, obj_images, sub_obj_union_images):
        """Extract the image feature vectors.

        Args:
            sub_images: The input subject image crop.
            obj_images: The input object image crop.
            sub_obj_union_images: The union of the input subject
                                  and input object
        Returns:
            The prediction scores for those regions.
        """
        sub_features = self.cnn(sub_images)
        obj_features = self.cnn(obj_images)
        union_features = self.cnn(sub_obj_union_images)
        all_features = torch.cat((sub_features, obj_features, union_features), 1)
        all_features = all_features.squeeze()
        scores = self.fc(all_features)
        return self.sigmoid(scores)


class PredictPredicateNN(nn.Module):
    """Model that predicts the relationship between a subject and an object"""

    def __init__(self, output_size):
        """Loads the pretrained DenseNet-121 model and replaces the last
           fully connected layer.
        """
        super(PredictPredicateNN, self).__init__()
        densenet = models.densenet121(pretrained=True)
        in_features = densenet.classifier.in_features
        modules = list(densenet.children())[:-1] # delete the last fc layer.
        modules.append(nn.AvgPool2d(7)) # add back the average pooling
        self.cnn = nn.Sequential(*modules)
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(3*in_features, output_size)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
        """
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, sub_images, obj_images, sub_obj_union_images):
        """Extract the image feature vectors.

        Args:
            sub_images: The input subject image crop.
            obj_images: The input object image crop.
            sub_obj_union_images: The union of the input subject
                                  and input object
        Returns:
            The prediction scores for those regions.
        """
        #features = self.cnn(images)
        sub_features = self.cnn(sub_images)
        obj_features = self.cnn(obj_images)
        union_features = self.cnn(sub_obj_union_images)
        all_features = torch.cat((sub_features, obj_features, union_features), 1)
        all_features = all_features.squeeze()
        scores = self.fc(all_features)
        return scores


