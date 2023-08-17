import torch
import time

class Trainer:
    def __init__(self, loader, model, criterion, optimizer, device, scheaduler=None):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheaduler = scheaduler
        self.device = device
    
    def train(self, epochs, log_interval, save_interval):
        self.model.train()
        self.model.to(self.device)

        times = []

        for i in range(epochs):
            st = time.time()

            batch = self.loader.get_batch()
            inputs, targets = batch['inputs'], batch['targets']
            inputs, targets = inputs.type(torch.int), targets.type(torch.float)
            inputs, targets  = inputs.to(self.device), targets.to(self.device)
            targets = targets.view(targets.size()[0], -1)
            logits = self.model(inputs)
            logits = logits.view(targets.size()[0], -1)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheaduler is not None:
                self.scheaduler.step()

            et = time.time()

            times.append((et - st))

            avg = torch.mean(torch.tensor(times))

            if i % log_interval == 0:
                print("Epoch {}, loss: {}, epoch time: {}, ETA: {}".format(i, loss, times[-1], epochs - i * avg))

            if i % save_interval == 0:
                name = "parameters\\params_iter_{}_loss_{}.params".format(i, int(loss))
                torch.save(self.model.state_dict(), name)
