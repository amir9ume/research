#credits : Meghana Moorthy Bhat "

class BertModel_pretrained(nn.Module):
    def __init__(self,
            text, 
            pretrain_path, 
            bert_model,
            ):
        super(BertModel_pretrained, self).__init__()
        self.model = None
        self.tokenized_text = []
        self.tokenizer = None

        if bert_model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        elif bert_model == 'scibert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

        if bert_model == 'bert':
            self.model = BertModel.from_pretrained(pretrain_path)
        elif bert_model == 'scibert':
            self.model = BertModel.from_pretrained(pretrain_path)
        else:
            self.model = BertModel.from_pretrained(pretrain_path)    

        for i in text:
            self.tokenized_text.append(self.tokenizer.tokenize(i))
        print("Tokenization from bert done")
        #tokens_tensor = []
        #segments_tensor = []
        

    def tokenize_tensor(self, device):
        tokens_tensor = []
        segments_tensor = []
        for token in self.tokenized_text:
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(token)
            segments_ids = [1] * len(token)

            # Convert inputs to PyTorch tensors
            token_tensor = torch.tensor([indexed_tokens])
            segment_tensor = torch.tensor([segments_ids])
            tokens_tensor.append(token_tensor.to(device))
            segments_tensor.append(segment_tensor.to(device))
        return tokens_tensor, segments_tensor

    def get_sentenceEmbedding(self, tokens_tensor, segments_tensor, length):
        if tokens_tensor.shape[1] > 0:
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensor)
            layer_i = 0
            batch_i = 0
            token_i = 0

            # Will have the shape: [# tokens, # layers, # features]
            token_embeddings = [] 

            # For each token in the sentence...
            for token_i in range(len(encoded_layers[layer_i][batch_i])):
              
              # Holds 12 layers of hidden states for each token 
              hidden_layers = [] 
              
              # For each of the 12 layers...
              for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]
                if vec is None:
                    print("No vector found in layer for token:", layer_i, token_i)
                hidden_layers.append(vec)
                
              token_embeddings.append(hidden_layers)

            # Sanity check the dimensions:
            print ("Number of tokens in sequence:", len(token_embeddings))
            print ("Number of layers per token:", len(token_embeddings[0]))
            #concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
            summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
            sentence_embedding = torch.mean(encoded_layers[11], 1)
            #print ("Our final sentence embedding vector of shape:", sentence_embedding.shape)
            return sentence_embedding

def decode(model, logits, vocab_size):
    """Return probability distribution over words."""
    logits_reshape = logits.view(-1, vocab_size)
    out_probs = F.softmax(logits_reshape)
    word_probs = out_probs.view(
        logits.size()[0], logits.size()[1], logits.size()[2]
    )
    return out_probs, word_probs