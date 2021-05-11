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


class DoubleCritic(nn.Module):
    """
    implementation of a critic network Q(s, a)
    """
    def __init__(self, state_dim, num_action, hidden_size1, hidden_size2):
        super(DoubleCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_action)

        self.fc4 = nn.Linear(state_dim, hidden_size1)
        self.fc5 = nn.Linear(hidden_size1, hidden_size2)
        self.fc6 = nn.Linear(hidden_size2, num_action)

    def forward(self, state):
        # given a state s, the network returns a vector Q(s,) of length |A|
        # network with two hidden layers
        x1 = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x2))
        x2 = self.fc3(x2)

        return x1, x2

    def Q1(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
