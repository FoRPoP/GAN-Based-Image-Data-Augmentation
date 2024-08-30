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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.hidden(x)
        x = self.output(x)
        
        return x

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

    def evaluate_model(self, test_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:

        self.eval()
        predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds)
                all_labels.append(labels)
        
        predictions = torch.cat(predictions)
        all_labels = torch.cat(all_labels)
        
        accuracy = (predictions == all_labels).float().mean().item()
        f1 = f1_score(all_labels.cpu(), predictions.cpu(), average='weighted')
        precision = precision_score(all_labels.cpu(), predictions.cpu(), average='weighted')
        recall = recall_score(all_labels.cpu(), predictions.cpu(), average='weighted')
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print("\nClassification Report:\n", classification_report(all_labels.cpu(), predictions.cpu(), digits=4))
        
        return all_labels.cpu(), predictions.cpu()

    def plot_confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
