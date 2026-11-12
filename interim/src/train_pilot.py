"""Explicitly condition-mean GEARS pilot, not published single-cell training.

Uses the unmodified upstream GEARS_Model class with train-only coexpression,
a target-gene-only GO graph, and 32 hidden units. Optional frozen GenePT features
are projected into the post-GO perturbation representation with a forward hook.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import pickle
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from baselines import correlation

ROOT=Path(__file__).resolve().parents[1]


def coexpression_graph(profiles, k=20, threshold=.4):
    z=profiles-profiles.mean(axis=0)
    z=z/np.maximum(np.linalg.norm(z,axis=0),1e-8)
    sources=[];targets=[];weights=[]
    for start in range(0,z.shape[1],256):
        corr=np.abs(z[:,start:start+256].T@z)
        cols=np.argpartition(corr,-(k+1),axis=1)[:,-(k+1):]
        rows=np.arange(len(cols))[:,None]
        vals=corr[rows,cols]
        sel=(vals>threshold)&(cols!=(rows+start))
        sources.extend(np.broadcast_to(rows+start,cols.shape)[sel].tolist())
        targets.extend(cols[sel].tolist());weights.extend(vals[sel].tolist())
    return torch.tensor([sources,targets],dtype=torch.long),torch.tensor(weights,dtype=torch.float32)


def go_graph(genes,gene2go):
    source=[];target=[];weight=[]
    for i,g in enumerate(genes):
        a=set(gene2go[g]);scores=[]
        for j,h in enumerate(genes):
            b=set(gene2go[h]);u=len(a|b)
            score=len(a&b)/u if u else 0
            if j!=i and score>.1:scores.append((score,j))
        for score,j in sorted(scores,reverse=True)[:20]:
            source.append(i);target.append(j);weight.append(score)
    return torch.tensor([source,target],dtype=torch.long),torch.tensor(weight,dtype=torch.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--embedding',action='store_true')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--hidden',type=int,default=32)
    parser.add_argument('--split',type=int,default=101)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    d=np.load(ROOT/'data/processed/condition_profiles.npz')
    means=d['means'];conditions=d['conditions'].tolist();idx={c:i for i,c in enumerate(conditions)}
    top=d['top1000'];ctrl=means[idx['ctrl']]
    split=next(s for s in json.loads((ROOT/'data/processed/splits.json').read_text()) if s['seed']==args.split)
    pert_genes=sorted({g for c in conditions for g in c.split('+') if g!='ctrl'})
    pertidx={g:i for i,g in enumerate(pert_genes)}
    train=[idx[c] for c in split['train']]
    graphfile=ROOT/f'data/processed/graphs_split{args.split}.pt'
    if graphfile.exists():graphs=torch.load(graphfile,map_location='cpu')
    else:
        gene2go=pickle.load(open(ROOT/'data/raw/gene2go_all.pkl','rb'))
        print('Constructing train-only condition-mean coexpression graph',flush=True)
        ce,cw=coexpression_graph(means[train])
        ge,gw=go_graph(pert_genes,gene2go)
        graphs=dict(ce=ce,cw=cw,ge=ge,gw=gw)
        torch.save(graphs,graphfile)
    print('Graph edges',len(graphs['cw']),len(graphs['gw']),flush=True)
    config=dict(num_genes=means.shape[1],num_perts=len(pert_genes),hidden_size=args.hidden,
                uncertainty=False,num_go_gnn_layers=1,num_gene_gnn_layers=1,decoder_hidden_size=16,
                no_perturb=False,G_coexpress=graphs['ce'],G_coexpress_weight=graphs['cw'],
                G_go=graphs['ge'],G_go_weight=graphs['gw'],device=args.device)
    spec=importlib.util.spec_from_file_location('upstream_gears_model',ROOT/'vendor/GEARS/gears/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.GEARS_Model(config).to(args.device)
    if args.embedding:
        emb=np.load(ROOT/'data/processed/genept_target_embeddings.npz')
        lookup={g:i for i,g in enumerate(emb['genes'])}
        features=np.stack([emb['embeddings'][lookup[g]] for g in pert_genes])
        features=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
        model.register_buffer('prior_features',torch.from_numpy(features).to(args.device))
        model.prior_projection=torch.nn.Sequential(torch.nn.Linear(features.shape[1],args.hidden),torch.nn.LayerNorm(args.hidden)).to(args.device)
        model.prior_scale=torch.nn.Parameter(torch.tensor(.2,device=args.device))
        model.sim_layers[-1].register_forward_hook(lambda _m,_inputs,out:out+model.prior_scale*model.prior_projection(model.prior_features))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)
    g=means.shape[1]
    def batch(indices):
        names=[conditions[i] for i in indices]
        perts=[[pertidx[p] for p in c.split('+')] if c!='ctrl' else [-1] for c in names]
        return SimpleNamespace(x=torch.from_numpy(np.tile(ctrl,(len(indices),1)).reshape(-1,1)).to(args.device),
                               batch=torch.arange(len(indices),device=args.device).repeat_interleave(g),pert_idx=perts)
    def predict(indices):
        model.eval();output=[]
        with torch.no_grad():
            for j in range(0,len(indices),4):output.append(model(batch(indices[j:j+4])).cpu().numpy())
        return np.concatenate(output)
    method='GEARS + GenePT pilot' if args.embedding else 'GEARS pilot'
    slug='gears_genept' if args.embedding else 'gears'
    best=np.inf;history=[];stale=0;started=time.time()
    checkpoint=ROOT/f'results/{slug}_split{args.split}_seed{args.seed}.pt'
    for epoch in range(args.epochs):
        model.train();losses=[];tick=time.time()
        for b in range(0,len(train),4):
            if b==0:order=rng.permutation(train)
            indices=order[b:b+4].tolist()
            if len(indices)==1:indices=indices*2
            optimizer.zero_grad(set_to_none=True)
            pred=model(batch(indices));truth=torch.from_numpy(means[indices]).to(args.device)
            loss=(pred-truth).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(loss.item())
        vi=[idx[c] for c in split['validation']]
        valpred=predict(vi)
        val=float(np.mean((valpred[:,top]-means[vi][:,top])**2))
        row=dict(epoch=epoch+1,training_mse=float(np.mean(losses)),validation_mse=val,seconds=time.time()-tick)
        history.append(row);print(method,row,flush=True)
        pd.DataFrame(history).to_csv(ROOT/f'results/{slug}_training_history.csv',index=False)
        if val<best:
            best=val;best_epoch=epoch+1;stale=0;torch.save(model.state_dict(),checkpoint)
        else:stale+=1
        if stale>=5:break
    model.load_state_dict(torch.load(checkpoint,map_location=args.device))
    ti=[idx[c] for c in split['test']];pred=predict(ti)
    rows=[]
    for i,p in zip(ti,pred):
        c=conditions[i];a,b=c.split('+');add=means[idx[a]]+means[idx[b]]-ctrl
        mse=float(np.mean((p[top]-means[i,top])**2));mseadd=float(np.mean((add[top]-means[i,top])**2))
        rows.append(dict(split=args.split,seed=args.seed,condition=c,method=method,mse=mse,
                         paired_improvement=mseadd-mse,pearson_delta=correlation(p[top]-ctrl[top],means[i,top]-ctrl[top])))
    pd.DataFrame(rows).to_csv(ROOT/f'results/{slug}_pair_metrics.csv',index=False)
    np.savez_compressed(ROOT/f'results/{slug}_predictions.npz',conditions=np.array(split['test']),predictions=pred)
    info=dict(method=method,split=args.split,seed=args.seed,epochs_completed=len(history),best_epoch=best_epoch,
              validation_mse=best,test_mse=float(np.mean([r['mse'] for r in rows])),wall_seconds=time.time()-started,
              hidden=args.hidden,readout_genes=g,parameters=sum(p.numel() for p in model.parameters()),device=args.device,
              training='MSE on condition-mean log-expression; not published single-cell training',
              graph='train-condition mean coexpression, top-20 abs correlations >0.4; GO Jaccard >0.1 top-20 among 100 target genes',
              upstream_commit=(ROOT/'vendor/GEARS/COMMIT').read_text().strip())
    (ROOT/f'results/{slug}_run.json').write_text(json.dumps(info,indent=2));print(info,flush=True)


if __name__=='__main__':main()
