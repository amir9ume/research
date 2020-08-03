"""pre-training for attention weights"""

import torch
import time
import datetime


from models import Match_LR, Regression_Attention_Over_docs
from utilities_model import get_fake_train_test_data_from_hidden_representations
from utilities_model import format_time, make_plot_training,prepare_data_bow,prepare_data,get_batch_eval, calculate_entropy_element, get_reviewer_entropy

#here you will do the routine where each reviewer is an expert on his own paper.
#then you will use these pre-trained attention weights,as hopefully this particular set of attention weights will know where to look at.


#load your fake data
data_path = '../../data_info/loaded_pickles_nips19/'
rep="LDA"
#manipulate your get train test data function. where for sub in their case, you pick up their own reviewer papers.
#and change bid values to 2 (or 3?)
train_sub,val_sub,test_sub, train_rev,val_rev,test_rev,y_train,y_val,y_test= get_fake_train_test_data_from_hidden_representations(rep,data_path)


batch_size=16
epochs=30
Attention_over_docs=True
criterion= torch.nn.MSELoss(reduction='sum')


"""maybe model for learning attention weights should be different here.
to make sure attention is doing the weight lifting, and not the Match-LR part"""

model_name="Match_LR"
model= Match_LR(batch_size,25,25,4,Attention_over_docs)
optimizer= torch.optim.SGD(model.parameters(), lr = 0.001,momentum=0.9)


#run the training regimen
losses= []
training_stats = []

for e_num in range(epochs):
    loss_ep = 0
    correct=0
    wrong=0
    # Measure how long the training epoch takes.
    t0 = time.time()
    
    model.train()
    for i in range(0, len(y_train), batch_size):
        mini_batch_submitted_paper, mini_batch_reviewer_paper, y = get_batch_eval(model_name,train_sub, train_rev, y_train, i, batch_size,model.padding) 
        if len(y.shape)>1:
            optimizer.zero_grad()
            if model_name=="Match_LR":
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            else:    
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            loss = criterion(prediction, y.argmax(dim=1).float())
            loss_ep += loss.item()
            loss.backward()         
            optimizer.step()
            class_label = torch.round(prediction).squeeze(dim=0)
            trg_label = y.argmax(dim=1)
            correct = correct + torch.sum(class_label==trg_label).item()
                    
    losses.append(loss_ep/len(y_train))
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("Epoch:", e_num, " Loss:", losses[-1], ": Train Accuracy:", correct/len(y_train), "  Training epcoh took: {:}".format(training_time))

    # #Validate after each epoch
    model.eval()
    with torch.no_grad():
        correct=0
        wrong=0
        loss_val=0
        rev_entropy_val=0
        for i in range(0, len(y_val)):
            if model_name=="Regression_Simple" and rep!="BOW":
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), torch.mean(val_rev[i].unsqueeze(dim=0),dim=1).float()).float()
            elif rep=="BOW":
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), val_rev[i].unsqueeze(dim=0).float()).float()
            elif rep=="LDA" and model_name=="Match_LR" and model.padding==False:
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), torch.mean(val_rev[i].unsqueeze(dim=0),dim=1).float()).float()[0]
            else:        
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), val_rev[i].unsqueeze(dim=0).float()).float()
            loss = criterion(prediction, y_val[i].argmax(dim=0).float())
            loss_val += loss.item()
            
            rev_entropy= get_reviewer_entropy(val_rev[i].unsqueeze(dim=0).float())
            rev_entropy_val+= rev_entropy

            class_label = torch.round(prediction).squeeze(dim=0)
            trg_label = y_val[i].argmax(dim=0)
            if rep=="LDA" and model_name=="Match_LR":
                if class_label==trg_label:
                    correct= correct+1
            else:      
                correct = correct + torch.sum(class_label==trg_label).item()
           
        print("Validation Loss:", loss_val/len(y_val), ": Validation Accuracy:", correct/len(y_val))
        print('reviewer entropy ', rev_entropy_val/len(y_val))
        print("========================================================")
        
        
        training_stats.append(
        {
            'epoch': e_num ,
            'Training Loss': losses[-1],
            'Valid. Loss': loss_val/len(y_val),
            'Valid. Accur.': correct/len(y_val),
            'Training Time': training_time
            }
    )


#save your attention weights
PATH=rep+'-'+model_name+'-flag_attn-'+str(Attention_over_docs)+'-epochs-'+str(epochs)+'-batch_size-'+str(batch_size)
path="pre_trained_weights-attention-"+PATH
#should I extract just the attention weights?
torch.save(model.state_dict(), path)