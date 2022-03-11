import logging, os, sys, time, torch, copy
from torch import nn, Tensor
import math
import fairseq
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.sequence_generator import EnsembleModel, SequenceGenerator
from tqdm import tqdm
from fairseq.data import data_utils
from typing import Dict, List, Optional
from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder
import argparse
from generate import *
from fairseq.search import Search, BeamSearch


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("fairseq")

def write_result(results, output_file):
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')

def get_generator(args, task):
    generator = TuningSequenceGenerator(
                models,
                task.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                search_strategy=TuningBeamSearch(task.target_dictionary, getattr(args, "ratio", 1.0)),
                # search_strategy=None
            )
    return generator

def get_baseline_generator(args, task):
    generator = SequenceGenerator(
                models,
                task.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                search_strategy=BeamSearch(task.target_dictionary),
            )
    return generator


@torch.no_grad()
def generate(all_lines, task, generator, batch_size):
    data_size = len(all_lines)
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    all_results = []
    for start_idx in tqdm(range(0, data_size, batch_size)):
        batch_lines = [line for line in all_lines[start_idx: min(start_idx + batch_size, data_size)]]
        batch_ids = [src_dict.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]

        lengths = torch.LongTensor([t.numel() for t in batch_ids])
        batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
        batch_dataset.left_pad_source = True
        batch = batch_dataset.collater(batch_dataset)
        batch = utils.apply_to_sample(lambda t: t.to(device), batch)

        translations = generator.generate(models, batch)
        results = []
        for id, hypos in zip(batch["id"].tolist(), translations):
            results.append((id, hypos))
        batched_hypos = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        all_results.extend([tgt_dict.string(hypos[0]['tokens']).replace('@@ ', '') for hypos in batched_hypos])
        # all_results.extend([[[tgt_dict.string(h['tokens']), h['positional_scores']] for h in hypos] for hypos in batched_hypos])
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='path to model file, e.g., /to/path/checkpoint_best.pt')
    parser.add_argument('--bin-data', type=str, required=True,
                        help='directory containing src and tgt dictionaries')
    parser.add_argument('--input-path', type=str, required=True,
                        help='path to eval file, e.g., /to/path/conll14.bpe.txt')
    parser.add_argument('--output-path', type=str, required=True,
                        help='path to output file, e.g., /to/path/conll14.pred.txt')
    parser.add_argument('--batch', type=int, default=None,
                        help='batch size')
    parser.add_argument('--beam', type=int, default=5,
                        help='beam size')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='penalty factor; '
                             '> 1.0, aggressive & recall-oriented; '
                             '< 1.0, conservative & precision-oriented.')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='do not use APD')

    cmd_args = parser.parse_args()

    cmd_args.checkpoint_path = os.path.expanduser(cmd_args.checkpoint_path)
    cmd_args.bin_data = os.path.expanduser(cmd_args.bin_data)
    cmd_args.input_path = os.path.expanduser(cmd_args.input_path)
    cmd_args.output_path = os.path.expanduser(cmd_args.output_path)

    models, args, task = load_model_ensemble_and_task(filenames=[cmd_args.checkpoint_path],
                                                    arg_overrides={'data': cmd_args.bin_data})

    device = torch.device('cuda')
    model = models[0].to(device).eval()

    if not cmd_args.baseline:
        generator = get_generator(cmd_args, task)
    else:
        generator = get_baseline_generator(cmd_args, task)

    with open(cmd_args.input_path, 'r') as f:
        bpe_sents = [l.strip() for l in f.readlines()]

    logger.info(f'Generate batch {cmd_args.batch}, beam {cmd_args.beam}, lambda {cmd_args.ratio}')
    remove_bpe_results = generate(bpe_sents, task, generator, cmd_args.batch)

    write_result(remove_bpe_results, cmd_args.output_path)