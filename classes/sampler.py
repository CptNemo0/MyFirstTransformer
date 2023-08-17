import torch
import torch.nn.functional as F
import tiktoken

class Sampler():
    def __init__(self, CONTEXT_LEN, MODEL, DEVICE):
        self.context_len = CONTEXT_LEN
        self.model = MODEL
        self.device = DEVICE
        #Additional tokens
        self.PAD_WORD       = "<PAD>"
        self.PAD_NUMBER     = 100265
        self.EOS_NUMBER     = 13 # "." = tiktoken.decode([13])
        
        #Tokenizer setup
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        self.encoder = tiktoken.Encoding(
            name="cl100k_my",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={**cl100k_base._special_tokens, self.PAD_WORD: self.PAD_NUMBER}
        )
    
    def prepare_prompt(self, text):
        encoded = self.encoder.encode(text)
        
        prompt = []
        
        if(len(encoded) == self.context_len):
            prompt = encoded
        elif(len(encoded) > self.context_len):
            split = len(encoded) - self.context_len 
            prompt = encoded[split:]
        elif(len(encoded) < self.context_len):
            diff = self.context_len - len(encoded)
            for _ in range(diff):
                prompt.append(self.PAD_NUMBER)
            prompt += encoded
        
        return prompt
    
    @torch.no_grad()
    def generate(self, text):
        prompt = self.prepare_prompt(text)
        prompt_copy = prompt.copy()
        self.answer = []
        
        self.model.to(self.device)
        prompt = torch.tensor([prompt], device = self.device)
            
        new_token = 0
            
        for i in range(25):
            output = self.model(prompt)
            output = output[:, [-1], :].view(100288)
            probabilities = F.softmax(output, dim=0)
            new_token = torch.multinomial(probabilities, 1)
            self.answer.append(new_token)
            prompt_copy.append(new_token)
            prompt_copy = prompt_copy[1:]
            prompt = prompt_copy.copy()
            prompt = torch.tensor([prompt], device = self.device)
            
            
        decoded_answer = self.encoder.decode(self.answer)
        return decoded_answer
            