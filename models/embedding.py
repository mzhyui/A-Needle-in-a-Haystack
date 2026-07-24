import torch


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.tanh(x)


class Embed(torch.nn.Module):
    def __init__(self):
        super(Embed, self).__init__()
        self.TransNet = TransformerNet()
        self.TransNet.load_state_dict(torch.load('./assets/step_fun.pkl', weights_only=True))

    def forward(self, pattern, mask):
        batch_size, channels, height, width = pattern.shape
        random_values = torch.rand_like(pattern).reshape(-1, 1)
        transformer_input = torch.cat(
            (random_values, pattern.reshape(-1, 1), mask.reshape(-1, 1)),
            dim=1,
        )
        transformed = self.TransNet(transformer_input)
        return transformed.view(batch_size, channels, height, width)
