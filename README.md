# TransE
An implementation of TransE with tensorflow.

TransE is proposed by Antoine Bordes, Nicolas Usunier and Alberto Garcia-Duran in 2013. The paper title is [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

# Dataset

The dataset WN18 and FB15k are orginally published by TransE paper and cand be download [here](https://everest.hds.utc.fr/doku.php?id=en:transe)

The data used here are from [here](https://github.com/thunlp/KB2E)

# Train and test

## Run 

to run the model with default parameter setting: 

--- 'python3 TransE.py'. 

If you run this code using python2 there will be some error because print setting '\t' is not supported by python2. 

##  change parameter setting

to change the parameter setting you can use 

--- 'python3 TransE.py --help' 

to see the optional arguments. 

## test

the default setting for testing is testing once with 300 triples after every 10 training iteration



