import torch
import esm
import pandas as pd
import os
import numpy as np


torch.cuda.set_device(7)
model, alphabet =  esm.pretrained.esm2_t30_150M_UR50D() # esm.pretrained.esm2_t6_8M_UR50D() #
model=model.cuda()
batch_converter = alphabet.get_batch_converter()
model.eval()  

dir_name='PDB14189_'

def read():
    proteins=open(f"{dir_name}/PDB14189_P.txt","r").readlines()
    res=[]
    for i in range(0,len(proteins),2):
        name=proteins[i][1:-1]
        seq=proteins[i+1]
        if name not in names:
            res.append([name,seq])
    return res

names=os.listdir(f"{dir_name}/feature")
names=[i.replace(".npy","") for i in names]
datas=read()
datas = sorted(datas, key=lambda x: len(x[1]))
seq_length=1200
for index,i in enumerate(datas):
    if len(i[1])<=seq_length:
        pass
        # padding_num=seq_length-len(i[1])
        # padding_seq='X'*padding_num
        # if padding_num>0:
        #     datas[index][1]=datas[index][1]+padding_seq
    else:
        datas[index][1]=datas[index][1][0:seq_length]

batch_size=2
for start in range(0,len(datas),batch_size):
    print(start,start+batch_size,len(datas))
    data=datas[start:start+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens=batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
    ans=0
    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        #print(seq)
        np.save(f"{dir_name}/feature/{data[ans][0]}",token_representations[ans].cpu().numpy())
        np.save(f"{dir_name}/map/{data[ans][0]}",attention_contacts[: tokens_len, : tokens_len].cpu().numpy())
        ans+=1

### 处理单个

names=os.listdir(f"{dir_name}/feature")
names=[i.replace(".npy","") for i in names]
datas=read()
datas = sorted(datas, key=lambda x: len(x[1]))
for index,i in enumerate(datas):
    if len(i[1])<=seq_length:
        pass
        # padding_num=seq_length-len(i[1])
        # padding_seq='X'*padding_num
        # if padding_num>0:
        #     datas[index][1]=datas[index][1]+padding_seq
    else:
        datas[index][1]=datas[index][1][0:seq_length]

batch_size=1
for start in range(0,len(datas),batch_size):
    print(start,start+batch_size,len(datas))
    data=datas[start:start+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens=batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
    ans=0
    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        #print(seq)
        np.save(f"{dir_name}/feature/{data[ans][0]}",token_representations[ans].cpu().numpy())
        np.save(f"{dir_name}/map/{data[ans][0]}",attention_contacts[: tokens_len, : tokens_len].cpu().numpy())
        ans+=1