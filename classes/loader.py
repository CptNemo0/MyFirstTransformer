import torch
import torch.nn.functional as F
  
class Loader():
    def __init__(self, data, batch_size, context_len, num_embeddings):
        assert len(data) >= batch_size
        assert len(data[0]) == context_len + 1
        
        self.data = data
        self.batch_size = batch_size
        self.context_len = context_len
        self.num_embeddings = num_embeddings
        
    def get_batch(self):
        idx = torch.randint(0, len(self.data), (self.batch_size,))
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        examples = [self.data[i] for i in idx]
        
        examples = torch.tensor(examples)

        inputs = examples[:, :self.context_len]
        inputs.to(dtype = torch.int64)
        
        targets = examples[:, 1:]
        targets.to(dtype = torch.int64)
        one_hot = F.one_hot(targets, num_classes=self.num_embeddings)
        
        sample = {'inputs': inputs, 'targets': one_hot}
        return sample