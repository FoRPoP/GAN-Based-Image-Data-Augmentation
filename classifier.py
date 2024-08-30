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

    def train_model(self, train_loader: DataLoader, validation_loader: DataLoader, num_epochs: int = 100) -> None:

        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            #self.writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

                val_labels, val_preds = self.evaluate_model(validation_loader)
                val_accuracy = (val_preds == val_labels).float().mean().item()
                val_f1 = f1_score(val_labels, val_preds, average='weighted')
                #self.writer.add_scalar('Accuracy/validation', val_accuracy * 100, epoch)
                #self.writer.add_scalar('F1 Score/validation', val_f1, epoch)

        #self.writer.close()
