"""Define your architecture here."""
import torch
#from models import SimpleNet
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.conv4 = nn.Conv2d(24, 16, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(5, 5))        
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        """Compute a forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def my_competition_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = MyNet()
    model = model.to(device)
   
   
    # For GPU - Colab Pro
    #model = model.cuda()
    #torch.backends.cudnn.benchmark = True
    
    # load your model using exactly this line (don't change it):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('checkpoints/competition_model.pt')['model'])
    else:
        model.load_state_dict(torch.load('checkpoints/competition_model.pt',map_location=torch.device('cpu'))['model'])
    
    return model
