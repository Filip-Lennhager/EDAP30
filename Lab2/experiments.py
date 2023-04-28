from torch import optim, nn
from infrastructure import Experiment
from model import YourFirstNet

class YourFirstCNN(Experiment):
    def init_model(self, n_labels, **kwargs):
        # TODO: Write your code here
        self.ckpt.model = YourFirstNet(n_labels)
        self.ckpt.optimizer = optim.Adam(self.ckpt.model.parameters(), weight_decay=1e-4)
        self.ckpt.criterion = nn.CrossEntropyLoss()

        #raise NotImplementedError()