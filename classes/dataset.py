import tiktoken
import random

class Dataset():
    def __init__(self, dataset_path, context_len):
        #Additional tokens
        self.PAD_WORD       = "<PAD>"
        self.PAD_NUMBER     = 100265
        
        #Tokenizer setup
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        self.encoder = tiktoken.Encoding(
            name="cl100k_my",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={**cl100k_base._special_tokens, self.PAD_WORD: self.PAD_NUMBER}
        )
        
        #Accesing data
        self.dataset_path = dataset_path
        self.text = ''
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        #Encoding data
        self.encoded_data = self.encoder.encode(self.text)
        
        #Preparing examples
        self.context_len = context_len
        padding     = [self.PAD_NUMBER for _ in range(self.context_len)]
        self.data = padding + self.encoded_data
        i : int = 0
        e : int = i + self.context_len + 1

        self.examples = []

        for _ in range(len(self.encoded_data)):
            example = self.data[i : e]
            self.examples.append(example)
            i += 1
            e += 1

        random.shuffle(self.examples)
        
        self.split_rate = (int)(len(self.examples) * 0.9)
        
        self.sets = {}
        self.sets['train'] = self.examples[:self.split_rate]
        self.sets['val']   = self.examples[self.split_rate:]
        
    def get_dataset(self, split = 'train'):
        return self.sets[split]