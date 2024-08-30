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
    def load_and_preprocess_data(self, data: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None, validation_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:

        if data is not None and labels is not None:
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            dataset = test_dataset = TensorDataset(data, labels)
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
            dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(validation_split * num_train))

        np.random.shuffle(indices)

        train_idx, validation_idx = indices[split:], indices[:split]
        train_sampler, validation_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(validation_idx)
        
        train_loader = DataLoader(dataset, batch_size=2048, sampler=train_sampler)    
        validation_loader = DataLoader(dataset, batch_size=1000, sampler=validation_sampler)
        test_loader = DataLoader(test_dataset, batch_size=1000)
        
        return train_loader, validation_loader, test_loader
