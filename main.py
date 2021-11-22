#!/usr/bin/env python
# coding: utf-8

# In[1]:


from copy import deepcopy
from pprint import pprint

import torch.cuda
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from baal import ActiveLearningDataset, ModelWrapper
from baal.active import ActiveLearningLoop
from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module
from baal.utils.metrics import Accuracy
import numpy as np


# In[2]:


use_cuda = torch.cuda.is_available()
NO_OF_INITIAL_LABELLED = 20
DROPOUT_RATE = 0.4
EPOCH = 50
BATCH_SIZE = 128
NO_OF_ITERATIONS = 200


# In[3]:


train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# In[4]:


train_ds = MNIST("/dataset_mnist", train=True, transform=train_transform, download=True)
test_ds = MNIST("/dataset_mnist", train=False, transform=test_transform, download=True)


# In[5]:


al_dataset = ActiveLearningDataset(train_ds, pool_specifics={"transform": test_transform})


# In[6]:


def label_uniformly(al_dataset, no_per_class=2):
    initial_dataset_size = al_dataset.labelled.shape[0]
    indices_to_be_labelled = set()
    
    for class_label in range(10):
        n_iter = 0
        while(n_iter < no_per_class):
            idx = np.random.choice(initial_dataset_size, 1)[0]
            selected_label = al_dataset.get_raw(idx)[1]
            if idx not in indices_to_be_labelled and selected_label == class_label:
                indices_to_be_labelled.add(idx)
                n_iter = n_iter + 1
    #print(indices_to_be_labelled)
    for elt in indices_to_be_labelled:
        #print(elt.item())
        al_dataset.label(elt.item())


# In[7]:


al_dataset = ActiveLearningDataset(train_ds, pool_specifics={"transform": test_transform})
label_uniformly(al_dataset)
#al_dataset.label_randomly(NO_OF_INITIAL_LABELLED)  # Start with 20 items labelled.


# In[8]:


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Dropout(p=0.25),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.Dropout(p=0.25),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Dropout(p=0.5),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1),
)


# In[9]:


model = patch_module(model)
if use_cuda:
    model = model.cuda()


# In[10]:


wrapper = ModelWrapper(model=model, criterion=nn.CrossEntropyLoss())
#wrapper.metrics = dict()
#wrapper.add_metric("accuracy", Accuracy)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[11]:


bald = BALD()

al_loop = ActiveLearningLoop(
    dataset=al_dataset,
    get_probabilities=wrapper.predict_on_dataset,
    heuristic=bald,
    ndata_to_label=10,  # We will label 100 examples per step.
    # KWARGS for predict_on_dataset
    iterations=100,  # 20 sampling for MC-Dropout
    batch_size=BATCH_SIZE,
    use_cuda=use_cuda,
    verbose=False,
)


# In[12]:


initial_weights = deepcopy(model.state_dict())


# In[ ]:


for step in range(NO_OF_ITERATIONS):
    model.load_state_dict(initial_weights)
    train_loss = wrapper.train_on_dataset(
        al_dataset, optimizer=optimizer, batch_size=BATCH_SIZE, epoch=EPOCH, use_cuda=use_cuda
    )
    test_loss = wrapper.test_on_dataset(test_ds, batch_size=BATCH_SIZE, use_cuda=use_cuda)

    pprint(
        {
            "dataset_size": len(al_dataset),
            #"train_loss": wrapper.metrics["train_loss"].value,
            #"test_loss": wrapper.metrics["test_loss"].value,
            "train_accuracy": wrapper.metrics['train_accuracy'].value,
            "test_accuracy": wrapper.metrics['test_accuracy'].value,
        }
    )
    flag = al_loop.step()
    if not flag:
        # We are done labelling! stopping
        break


# In[ ]:


torch.save(wrapper.model.state_dict(), 'model_mnist.pth')

