"""Cold-start profile preparation without touching the interim report files."""
import json
import hashlib
import io
import pickle
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from gpu.common import INTERIM
from gpu.prepare import prepare_cells, prepare_profiles, download, FALLBACK_URLS


class PreparationTests(unittest.TestCase):
    def test_403_fallback_is_verified_and_keeps_source_on_rerun(self):
        payload = b'official data bytes'
        name = 'essential_all_data_pert_genes.pkl'
        primary = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
        def request(url, **kwargs):
            if url == primary:
                raise urllib.error.HTTPError(url, 403, 'Forbidden', None, None)
            response = io.BytesIO(payload)
            response.status = 200
            return response
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / name
            with patch('gpu.prepare.urllib.request.urlopen', side_effect=request) as opener, \
                 patch('gpu.prepare.OFFICIAL_SHA256', {name: hashlib.sha256(payload).hexdigest()}):
                first = download(path, primary)
                second = download(path, primary)
            self.assertEqual(opener.call_count, 2)
            self.assertEqual(first['url'], FALLBACK_URLS[name][0])
            self.assertEqual(second['url'], first['url'])
            self.assertEqual(path.read_bytes(), payload)

    def test_download_rejects_wrong_hash_before_persisting(self):
        name = 'essential_all_data_pert_genes.pkl'
        response = io.BytesIO(b'incorrect bytes')
        response.status = 200
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / name
            with patch('gpu.prepare.urllib.request.urlopen', return_value=response):
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    download(path, 'https://example.test/data')
            self.assertFalse(path.exists())

    def test_fresh_profiles_and_cell_cache_preserve_source_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'interim'
            cache = Path(d) / 'cache'
            for sub in ['data/raw', 'data/processed', 'src']:
                (root / sub).mkdir(parents=True)
            cache.mkdir()
            shutil.copyfile(INTERIM / 'src/prepare_data.py', root / 'src/prepare_data.py')
            targets = list('ABCDEFGH')
            conditions = sorted(targets + ['A+' + g for g in targets[1:7]] + ['ctrl'])
            genes = targets + [f'readout{i}' for i in range(24)]
            rng = np.random.default_rng(71)
            x = rng.random((len(conditions) * 4, len(genes)), dtype=np.float32)
            x[x < .6] = 0
            labels = np.repeat(conditions, 4)
            a = ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame({'condition': labels},
                           index=[f'cell{i}' for i in range(len(x))]),
                           var=pd.DataFrame({'gene_name': genes}, index=genes))
            a.write_h5ad(root / 'data/raw/norman.h5ad')
            with (root / 'data/raw/gene2go_all.pkl').open('wb') as f:
                pickle.dump({g: {'GO:00001'} for g in targets}, f)
            pairs = np.array([c for c in conditions if '+' in c])
            singles = [c for c in conditions if '+' not in c]
            splits = []
            for seed in [101, 202, 303, 404, 505]:
                perm = np.random.default_rng(seed).permutation(pairs)
                ntr = len(pairs) // 2
                nval = (len(pairs) - ntr) // 2
                splits.append(dict(seed=seed, train=singles + perm[:ntr].tolist(),
                                   validation=perm[ntr:ntr+nval].tolist(), test=perm[ntr+nval:].tolist()))
            source_splits = root / 'data/processed/splits.json'
            original = json.dumps(splits, indent=2)
            source_splits.write_text(original)
            with patch('gpu.prepare.INTERIM', root):
                prepare_profiles(cache)
                prepare_cells(cache)
            self.assertEqual(source_splits.read_text(), original)
            self.assertFalse((root / 'data/processed/condition_profiles.npz').exists())
            self.assertEqual(json.loads((cache / 'splits.json').read_text()), splits)
            np.testing.assert_array_equal(np.load(cache / 'cells.npy'), x)
            profiles = np.load(cache / 'profiles.npz')
            for c, mean in zip(profiles['conditions'], profiles['means']):
                np.testing.assert_allclose(mean, x[labels == c].mean(axis=0), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
