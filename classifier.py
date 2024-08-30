class MNISTClassifier(nn.Module):
    def __init__(self, lr: float = 0.0002, input_dim: int = 784, output_dim: int = 10, hidden_dim: int = 300, reg: float = 0.0001, dropout_rate: float = 0.1) -> None:

        super(MNISTClassifier, self).__init__()
        
        self.hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout_rate))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        #self.writer = SummaryWriter(log_dir='./logs')
