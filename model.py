import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    implementation of a critic network Q(s, a)
    """
    def __init__(self, state_dim, num_action, hidden_size1, hidden_size2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_action)

    def forward(self, state):
        # given a state s, the network returns a vector Q(s,) of length |A|
        # network with two hidden layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
