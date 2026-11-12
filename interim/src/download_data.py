"""Download the specific public releases used here, retaining their hashes.

No API keys are needed. Large files: ~0.6 GB Norman zip, ~2.2 GB extracted
AnnData, and ~0.6 GB GenePT zip. The GenePT pickle is from its authors' archive.
"""
from pathlib import Path
import hashlib
import json
import pickle
import shutil
import urllib.request
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'norman_scfoundation.zip':'https://api.figshare.com/v2/file/download/44477939',
    'gene2go_all.pkl':'https://raw.githubusercontent.com/biomap-research/scFoundation/aa51ac12ce3c9fd13bc7f918cf70f5f12e10cd37/GEARS/data/gene2go.pkl',
    'GenePT_embeddings_v2.zip':'https://zenodo.org/api/records/10833191/files/GenePT_emebdding_v2.zip/content',
}


def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in SOURCES.items():
        p=raw/name
        if not p.exists():
            temp=p.with_suffix(p.suffix+'.part')
            with urllib.request.urlopen(url,timeout=120) as r,temp.open('wb') as f:
                if r.status!=200:raise RuntimeError(f'Download returned HTTP {r.status}: {url}')
                shutil.copyfileobj(r,f,1024*1024)
            temp.replace(p)
        if p.stat().st_size<1000:raise RuntimeError(f'Empty or invalid download: {name}')
        sha=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):sha.update(chunk)
        records.append(dict(name=name,url=url,bytes=p.stat().st_size,sha256=sha.hexdigest()))
    h5=raw/'norman.h5ad'
    if not h5.exists():
        with zipfile.ZipFile(raw/'norman_scfoundation.zip') as z:
            member=next(n for n in z.namelist() if n.endswith('.h5ad'))
            with z.open(member) as src,h5.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    (ROOT/'results/source_manifest.json').write_text(json.dumps(records,indent=2))
    print('Verified public downloads and wrote source_manifest.json')


def extract_genept():
    d=np.load(ROOT/'data/processed/condition_profiles.npz')
    genes=sorted({g for c in d['conditions'] for g in c.split('+') if g!='ctrl'})
    with zipfile.ZipFile(ROOT/'data/raw/GenePT_embeddings_v2.zip') as z:
        with z.open('GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle') as f:emb=pickle.load(f)
    first=np.asarray(next(iter(emb.values())))
    values=np.stack([np.asarray(emb[g],dtype='float32') if g in emb else np.zeros_like(first,dtype='float32') for g in genes])
    present=np.array([g in emb for g in genes])
    np.savez_compressed(ROOT/'data/processed/genept_target_embeddings.npz',genes=np.array(genes),embeddings=values,present=present)
    info=dict(source='Zenodo 10833191, GenePT_gene_embedding_ada_text.pickle',variant='NCBI gene descriptions, text-embedding-ada-002',
              dimension=int(values.shape[1]),target_genes=len(genes),covered=int(present.sum()),missing=[g for g,m in zip(genes,present) if not m])
    (ROOT/'results/embedding_coverage.json').write_text(json.dumps(info,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--extract-genept',action='store_true');args=parser.parse_args()
    if args.extract_genept:extract_genept()
    else:main()
