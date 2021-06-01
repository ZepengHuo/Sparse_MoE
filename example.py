# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#

from moe import MoE
import torch
from torch import nn
from torch.optim import Adam


def train(x,y, model, loss_fn, optim):
    model.train()
    
    ## 1. forward propagation
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    
    ## 2. loss calculation
    # calculate prediction loss
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
        
    #initialization
    optim.zero_grad()

    ## 3. backward propagation
    total_loss.backward()
    
    ## 4. weight optimization
    optim.step()

    print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model

def eval(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float(), train=False)
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))



def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)

    # dummy target
    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x,y



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# arguments
input_size = 1000
num_classes = 20
num_experts = 10
hidden_size = 64
batch_size = 5
k = 4  #k: an integer - how many experts to use for each batch element
epochs = 10


# instantiate the MoE layer
model = MoE(input_size, num_classes, num_experts,hidden_size, k=k, noisy_gating=True)

model.to(device)



loss_fn = nn.NLLLoss()
optim = Adam(model.parameters())

x_train, y_train = dummy_data(batch_size, input_size, num_classes)
x_eval, y_eval = dummy_data(batch_size, input_size, num_classes)

x_train, y_train = x_train.to(device), y_train.to(device)
x_eval, y_eval = x_eval.to(device), y_eval.to(device)


#x_train, y_train = x_train.to(device), y_train.to(device)

for epoch in range(1, epochs):  ## run the model for 10 epochs
    # train
    model = train(x_train, y_train, model, loss_fn, optim)
    # evaluate
    eval(x_eval, y_eval, model, loss_fn)
