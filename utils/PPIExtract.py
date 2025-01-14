import torch
import esm
torch.cuda.set_device(1)

proteins=open("PP/protein.dictionary.tsv","r").read().split("\n")
proteins=[i.split("\t") for i in proteins]

for i in range(len(proteins)):
    if len(proteins[i][1])>1200:
        proteins[i][1]=proteins[i][1][0:1200]
    
lengths=[len(i[1]) for i in proteins]
lengths.sort()

#print(lengths)
# # Load ESM-2 model
model, alphabet =  esm.pretrained.esm2_t33_650M_UR50D()
model=model.cuda()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

import numpy as np

for index in range(len(proteins)):
    #print(start)
    data=proteins[index]
    data=[data]
    #print(data)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens=batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]
    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        #print(seq)
        np.save(f"PP/feature/{proteins[index][0]}",token_representations[0].cpu().numpy())
        np.save(f"PP/map/{proteins[index][0]}",attention_contacts[: tokens_len, : tokens_len].cpu().numpy())
       
    print(int(index/len(proteins)*100),"%")
