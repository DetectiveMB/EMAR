import logging
import os

import hydra
from indxr import Indxr
from omegaconf import DictConfig

from ranx import Run, Qrels, compare, fuse, optimize_fusion
from model.utils import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_dir = cfg.dataset.data_dir
    runs_dir = cfg.dataset.runs_dir

    logging_dir = cfg.logging.logging_dir
    logging_file = os.path.join(logging_dir, '4_fuse.log')
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)


    split = 'val'
    val_neural_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}.lz4'))
    val_queries = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    val_bm25_runs = {q['id']: dict(zip(q['bm25_doc_ids'], q['bm25_doc_scores'])) for q in val_queries}
    val_bm25_runs = Run(val_bm25_runs)
    val_qrels = Qrels.from_file(os.path.join(data_dir, split, 'qrels.json'))

    best_params = optimize_fusion(
        qrels=val_qrels,
        runs=[val_bm25_runs, val_neural_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@100",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    logging.info(f'best parameters: {best_params[0]}')
    logging.info(f'best parameters results:\n {best_params[1]}')


    split = 'test'
    test_neural_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}.lz4'))
    test_neural_run.name = 'Neural'
    test_queries = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    test_bm25_runs = {q['id']: dict(zip(q['bm25_doc_ids'], q['bm25_doc_scores'])) for q in test_queries}
    test_bm25_runs = Run(test_bm25_runs)
    test_bm25_runs.name = 'BM25'
    test_qrels = Qrels.from_file(os.path.join(data_dir, split, 'qrels.json'))

    combined_test_run = fuse(
        runs=[test_bm25_runs, test_neural_run],
        norm="min-max",
        method="wsum",
        params=best_params[0],
    )
    combined_test_run.name = 'Neural + BM25'

    models = [
        test_neural_run,
        test_bm25_runs,
        combined_test_run
    ]
    report = compare(
        qrels=test_qrels,
        runs=models,
        metrics=['map@100', 'precision@1', 'mrr@3', 'ndcg@3', 'mrr@5', 'ndcg@5', 'mrr@10', 'ndcg@10'],
        max_p=0.01  # P-value threshold, 3 tests
    )
    logging.info(f'\n{report}')
    print(report)


if __name__ == '__main__':
    main()