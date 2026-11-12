import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from gpu.common import INTERIM, load_checkpoint
from gpu.data import CellData
from gpu.model import PerturbationModel, make_batch, gears_loss, upstream
from gpu.prepare import coexpression, exclude_k562, top_go_edges
from gpu import train
from gpu.run_suite import jobs


def fixture(path):
    rng = np.random.default_rng(912)
    conditions = ['A', 'A+B', 'A+C', 'B', 'B+C', 'C', 'ctrl']
    labels = np.repeat(np.arange(len(conditions)), 4)
    cells = rng.random((len(labels), 8), dtype=np.float32)
    means = np.stack([cells[labels == i].mean(axis=0) for i in range(len(conditions))])
    genes = np.array(list('ABCDEFGH'))
    np.save(path / 'cells.npy', cells)
    np.save(path / 'labels.npy', labels)
    np.savez(path / 'profiles.npz', means=means, conditions=np.array(conditions), genes=genes, top1000=np.arange(8))
    np.save(path / 'top20.npy', np.tile(np.arange(3), (7, 1)))
    (path / 'cells.json').write_text(json.dumps(dict(conditions=conditions, genes=genes.tolist(), raw_sha256='fixture')))
    (path / 'splits.json').write_text(json.dumps([dict(seed=101, train=['A', 'B', 'C', 'ctrl', 'A+B'],
                                                    validation=['A+C'], test=['B+C'])]))
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    np.savez(path / 'graphs.npz', pert_genes=np.array(list('ABCDE')), ce=edges, cw=np.ones(4, np.float32),
             ge=edges, gw=np.ones(4, np.float32))
    for variant in ['genept', 'depmap']:
        np.savez(path / f'{variant}.npz', genes=np.array(list('ABCDE')), features=rng.random((5, 7), dtype=np.float32))
    return CellData(path, 101)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.data = fixture(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_matches_unmodified_upstream(self):
        args = self.data.model_args(8, 'cpu')
        torch.manual_seed(12)
        reference = upstream.GEARS_Model(args).eval()
        model = PerturbationModel(args).eval()
        model.core.load_state_dict(reference.state_dict())
        batch = make_batch(np.tile(self.data.control, (3, 1)), [[0], [0, 1], [-1]], 'cpu')
        with torch.no_grad():
            torch.testing.assert_close(model(batch), reference(batch), rtol=0, atol=0)

    def test_loss_matches_upstream_value_and_gradient(self):
        tree = ast.parse((INTERIM / 'vendor/GEARS/gears/utils.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_fct')
        scope = dict(torch=torch, np=np)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<upstream loss>', 'exec'), scope)
        torch.manual_seed(5)
        p = torch.randn(4, 8, requires_grad=True)
        q = p.detach().clone().requires_grad_()
        y = torch.randn_like(p)
        names = ['A', 'A', 'ctrl', 'B']
        masks = self.data.masks('cpu')
        c = torch.tensor(self.data.control)
        actual = gears_loss(p, y, names, c, masks)
        expected = scope['loss_fct'](q, y, names, ctrl=c, dict_filter=masks, direction_lambda=.1)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(p.grad, q.grad)

    def test_attention_permutation_and_prior_gradients(self):
        model = PerturbationModel(self.data.model_args(8, 'cpu'), self.data.prior('genept'), 'attention').eval()
        x = np.tile(self.data.control, (2, 1))
        a = model(make_batch(x, [[0, 1], [2]], 'cpu'))
        b = model(make_batch(x, [[1, 0], [2]], 'cpu'))
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        a.sum().backward()
        self.assertIsNotNone(model.prior_projection[0].weight.grad)
        self.assertGreater(model.set_attention.self_attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertFalse(model.prior.requires_grad)

    def test_exact_control_mean_inference(self):
        model = PerturbationModel(self.data.model_args(8, 'cpu')).eval()
        rows = self.data.control_rows
        with torch.no_grad():
            individual = np.stack([model(make_batch(self.data.cells[r:r+1].copy(), [[0, 1]], 'cpu')).numpy()[0] for r in rows])
        average = train.predict_means(model, self.data, ['A+B'], 'cpu')[0]
        np.testing.assert_allclose(average, individual.mean(axis=0), rtol=1e-5, atol=1e-6)

    def test_cell_graph_matches_numpy_and_ignores_pairs(self):
        ce, cw = coexpression(self.data.cells, self.data.labels, self.data.conditions, 'cpu', k=2, threshold=.01)
        rows = np.isin(self.data.labels, [i for i, c in enumerate(self.data.conditions) if '+' not in c])
        corr = np.abs(np.corrcoef(self.data.cells[rows].T))
        for source, target, weight in zip(*ce, cw):
            self.assertAlmostEqual(weight, corr[source, target], places=5)
        changed = self.data.cells.copy()
        changed[~rows] = 10000
        other, weights = coexpression(changed, self.data.labels, self.data.conditions, 'cpu', k=2, threshold=.01)
        np.testing.assert_array_equal(ce, other)
        np.testing.assert_allclose(cw, weights)
        self.assertEqual(set(self.data.masks('cpu')), set(self.data.split['train']))

    def test_official_go_topk_selection(self):
        frame = pd.DataFrame(dict(source=['A', 'B', 'C', 'A', 'B', 'C'], target=['A']*3+['B']*3,
                                  importance=[1., .8, .8, .2, 1., .9]))
        path = self.path / 'go.csv'
        frame.to_csv(path, index=False)
        edge, weights = top_go_edges(path, ['A', 'B', 'C'], k=1)
        expected = frame.groupby('target', sort=True).apply(lambda f: f.nlargest(2, 'importance'), include_groups=False)
        np.testing.assert_allclose(weights, expected.importance)
        np.testing.assert_array_equal(edge[0], [0, 1, 1, 2])

    def test_depmap_excludes_k562_and_derivatives(self):
        frame = pd.DataFrame({'A': [1., 2., 3., 4.]}, index=['ACH-000551', 'derived', 'detroit', 'other'])
        metadata = pd.DataFrame(dict(ModelID=frame.index, CellLineName=['K-562', 'P2UR/K-562', 'Detroit 562', 'Other']))
        kept, excluded = exclude_k562(frame, metadata)
        self.assertEqual(excluded, ['ACH-000551', 'derived'])
        self.assertEqual(kept.index.tolist(), ['detroit', 'other'])
        with self.assertRaises(ValueError):
            exclude_k562(frame, metadata.iloc[:1])

    def args(self, name):
        return train.parser().parse_args(['--cache', str(self.path), '--output', str(self.path / name),
                                         '--epochs', '2', '--hidden', '8', '--batch-size', '4', '--device', 'cpu',
                                         '--threads', '2', '--prior', 'genept', '--aggregation', 'attention'])

    def test_resume_mid_epoch_matches_uninterrupted(self):
        self.assertEqual(train.run(self.args('continuous')), 0)
        real_loss = train.gears_loss
        def stop_after_update(*args, **kwargs):
            result = real_loss(*args, **kwargs)
            train.STOP = True
            return result
        with patch.object(train, 'gears_loss', stop_after_update):
            self.assertEqual(train.run(self.args('resumed')), 75)
        checkpoint = load_checkpoint(self.path / 'resumed/latest.pt')
        self.assertEqual(checkpoint['updates'], 1)
        self.assertEqual(train.run(self.args('resumed')), 0)
        full = load_checkpoint(self.path / 'continuous/latest.pt')
        resumed = load_checkpoint(self.path / 'resumed/latest.pt')
        for key, value in full['model'].items():
            torch.testing.assert_close(value, resumed['model'][key], rtol=0, atol=0)
        np.testing.assert_array_equal(np.load(self.path / 'continuous/predictions.npz')['predictions'],
                                      np.load(self.path / 'resumed/predictions.npz')['predictions'])
        self.assertEqual(len(full['history']), 2)
        self.assertEqual(full['best_epoch'], resumed['best_epoch'])

    def test_suite_covers_six_configs_five_splits_two_seeds(self):
        self.assertEqual(len(list(jobs('upstream'))), 60)
        self.assertEqual(len(list(jobs('extended'))), 70)


if __name__ == '__main__':
    unittest.main()
