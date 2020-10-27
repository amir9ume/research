import torch
import numpy as np
import torch.nn.functional as F
'''
credits:
simCLR for pytorch
https://github.com/sthalles/SimCLR
'''

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

def loss_fn(inp, targ):
    thau = 1.
    # Similarities
    inp_norm = inp / inp.norm(dim=1)[:,None]
    simils = torch.mm(inp_norm, inp_norm.transpose(0,1))
    # Good and Bad
    N = len(inp_norm)
    eye = (2*torch.eye(N)-1).to(device)
    eye[torch.arange(1,N, step=2), torch.arange(0,N, step=2)] = 1
    eye[torch.arange(0,N, step=2), torch.arange(1,N, step=2)] = 1
    #
    exps = torch.exp(simils)
    num = ((exps/thau)*(eye+1)/2).sum(dim=1)
    den = ((exps/thau)*(-eye+1)/2).sum(dim=1)
    loss = (-torch.log(num/den)).mean()
    return loss

#credits
# https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c
def loss_towards_ds(a,b,tau):
#    a_norm = torch.norm(a,dim=(1).reshape(-1,1)
    a_norm = torch.norm(a,dim=(1,2)).unsqueeze(dim=1).unsqueeze(dim=1)

    a_cap = torch.div(a,a_norm)
#    b_norm = torch.norm(b,dim=1).reshape(-1,1)
    b_norm = torch.norm(b,dim=(1,2)).unsqueeze(dim=1).unsqueeze(dim=1)
    b_cap = torch.div(b,b_norm) 
    #matrix of unit vectors
    a_cap_b_cap = torch.cat([a_cap,b_cap],dim=0)
#    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    a_cap_b_cap_transpose = (a_cap_b_cap).permute(2,0,1)
    b_cap_a_cap = torch.cat([b_cap,a_cap],dim=0)
    #similarity matrix
    sim = torch.bmm(a_cap_b_cap,a_cap_b_cap_transpose)
    
    sim_by_tau = torch.div(sim,tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)

    #check why dim=1. you want to store across all batch examples, 2N
    #looks like denominator
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=0)
    
    #diag returns a vector(diagonal) if input is matrix
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap,b_cap_a_cap),tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators,denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)

#credits https://www.egnyte.com/blog/2020/07/understanding-simclr-a-framework-for-contrastive-learning/
class ContrastiveLossELI5(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss

class ContrastiveLoss(torch.nn.Module):
   def __init__(self, batch_size, temperature=0.5):
       super().__init__()
       self.batch_size = batch_size
       self.register_buffer("temperature", torch.tensor(temperature))
       self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
          

   def forward(self, emb_i, emb_j):
       """
       emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
       z_i, z_j as per SimCLR paper
       """
       z_i = F.normalize(emb_i, dim=1)
       z_j = F.normalize(emb_j, dim=1)

       representations = torch.cat([z_i, z_j], dim=0)
       similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
      
       sim_ij = torch.diag(similarity_matrix, self.batch_size)
       sim_ji = torch.diag(similarity_matrix, -self.batch_size)
       positives = torch.cat([sim_ij, sim_ji], dim=0)
      
       nominator = torch.exp(positives / self.temperature)
       denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
  
       loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(loss_partial) / (2 * self.batch_size)
       return loss
