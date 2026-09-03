import json
import logging
import os

import hydra
import torch
from tqdm import tqdm
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
    logging_file = os.path.join(logging_dir, '14_fuse.log')
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)


    split = 'val'
    val_all_run = Run.from_file(os.path.join(cfg.general.all_runs_dir, cfg.dataset.name, split, f'{cfg.model.name}.lz4'))
    val_single_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}.lz4'))
    val_teacher_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.teacher_name}.lz4'))
    val_moe_kd_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe.lz4'))
    val_moe_top_k_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe_top_k.lz4'))
    val_moe_rand_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe_rand.lz4'))
    
    val_queries = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    val_bm25_runs = {q['id']: dict(zip(q['bm25_doc_ids'], q['bm25_doc_scores'])) for q in val_queries}
    val_bm25_runs = Run(val_bm25_runs)
    val_qrels = Qrels.from_file(os.path.join(data_dir, split, 'qrels.json'))

    best_single_params = optimize_fusion(
        qrels=val_qrels,
        runs=[val_bm25_runs, val_single_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )
    best_teacher_params = optimize_fusion(
        qrels=val_qrels,
        runs=[val_bm25_runs, val_teacher_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )
    best_all_params = optimize_fusion(
        qrels=val_qrels,
        runs=[val_bm25_runs, val_all_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )
    best_kd_params = optimize_fusion(
        qrels=val_qrels,
        runs=[val_bm25_runs, val_moe_kd_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )
    best_top_k_params = optimize_fusion(
        qrels=val_qrels,
        runs=[val_bm25_runs, val_moe_top_k_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )
    best_rand_params = optimize_fusion(
        qrels=val_qrels,
        runs=[val_bm25_runs, val_moe_rand_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    split = 'test'
    test_all_run = Run.from_file(os.path.join(cfg.general.all_runs_dir, cfg.dataset.name, split,f'{cfg.model.name}.lz4'))
    test_teacher_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.teacher_name}.lz4'))
    test_single_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}.lz4'))
    test_moe_kd_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe.lz4'))
    test_moe_top_k_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe_top_k.lz4'))
    test_moe_rand_run = Run.from_file(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe_rand.lz4'))
    
    test_queries = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    test_bm25_runs = {q['id']: dict(zip(q['bm25_doc_ids'], q['bm25_doc_scores'])) for q in test_queries}
    test_bm25_runs = Run(test_bm25_runs)
    test_bm25_runs.name = 'BM25'
    test_qrels = Qrels.from_file(os.path.join(data_dir, split, 'qrels.json'))

    combined_teacher_run = fuse(
        runs=[test_bm25_runs, test_teacher_run],
        norm="min-max",
        method="wsum",
        params=best_teacher_params[0],
    )
    combined_teacher_run.name = 'TEACHER'

    combined_single_run = fuse(
        runs=[test_bm25_runs, test_single_run],
        norm="min-max",
        method="wsum",
        params=best_single_params[0],
    )
    combined_single_run.name = 'SINGLE'

    combined_all_run = fuse(
        runs=[test_bm25_runs, test_all_run],
        norm="min-max",
        method="wsum",
        params=best_all_params[0],
    )
    combined_all_run.name = 'ALL'
    
    combined_kd_run = fuse(
        runs=[test_bm25_runs, test_moe_kd_run],
        norm="min-max",
        method="wsum",
        params=best_kd_params[0],
    )
    combined_kd_run.name = 'KD'

    combined_top_k_run = fuse(
        runs=[test_bm25_runs, test_moe_top_k_run],
        norm="min-max",
        method="wsum",
        params=best_top_k_params[0],
    )
    combined_top_k_run.name = 'TOP_1'

    combined_rand_run = fuse(
        runs=[test_bm25_runs, test_moe_rand_run],
        norm="min-max",
        method="wsum",
        params=best_rand_params[0],
    )
    combined_rand_run.name = 'RND'
    


    models = [
        combined_teacher_run,
        combined_single_run,
        combined_all_run,
        combined_kd_run,
        combined_top_k_run,
        combined_rand_run
    ]
    report = compare(
        qrels=test_qrels,
        runs=models,
        metrics=['map@100', 'mrr@3', 'ndcg@3', 'mrr@10', 'ndcg@10'],
        max_p=0.01  # P-value threshold, 3 tests
    )
    logging.info(f'\n{report}')
    print(report)

    
if __name__ == '__main__':
    main()