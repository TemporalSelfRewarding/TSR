"""Regression coverage for the official-code fixes; no model download or GPU required."""
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

from datasets import Dataset, load_dataset
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Load real production modules without __init__'s unrelated HumanEval network fetch.
PACKAGE = '_tsr_regression_alignment'
module = types.ModuleType(PACKAGE)
module.__path__ = [str(ROOT / 'alignment-handbook/src/alignment')]
sys.modules[PACKAGE] = module
data_module = importlib.import_module(PACKAGE + '.data')
TSRDPOTrainer = importlib.import_module(PACKAGE + '.dpo_trainer').TSRDPOTrainer


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tsr regression ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class DataLoadingTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.data = self.root / 'data'
        self.rows = [{'prompt': f'q{i}', 'chosen': 'good', 'rejected': 'bad'} for i in range(3)]
        write_rows(self.data / 'train.json', self.rows)

    def load(self):
        return data_module.get_datasets({str(self.data): 1.0}, splits=['train'],
                                       columns_to_keep=['prompt', 'chosen', 'rejected'],
                                       shuffle=False, local_train_file='train.json')['train']

    def test_nested_official_duplicate_is_loaded_once(self):
        write_rows(self.data / 'official/train.json', self.rows)
        self.assertEqual(len(load_dataset(str(self.data), split='train')), 6)
        self.assertEqual(list(self.load()), self.rows)
        self.assertTrue((self.data / 'official/train.json').exists())

    def test_unrelated_json_is_excluded(self):
        write_rows(self.data / 'train_extra.json', [{'unrelated': 'must not be loaded'}])
        self.assertEqual(list(self.load()), self.rows)

    def test_missing_canonical_file_fails(self):
        (self.data / 'train.json').unlink()
        write_rows(self.data / 'official/train.json', self.rows)
        with self.assertRaisesRegex(FileNotFoundError, 'Canonical DPO'):
            self.load()

    def test_sft_keeps_both_input_files_and_test_split(self):
        (self.data / 'train.json').unlink()
        write_rows(self.data / 'ift_train.jsonl', self.rows[:2])
        write_rows(self.data / 'eft_train.jsonl', self.rows[2:])
        write_rows(self.data / 'ultrachat_test.json', self.rows[:1])
        loaded = data_module.get_datasets({str(self.data): 1.0}, splits=['train', 'test'],
                                          columns_to_keep=['prompt', 'chosen', 'rejected'], shuffle=False)
        self.assertCountEqual(list(loaded['train']), self.rows)
        self.assertEqual(len(loaded['test']), 1)


class PairTests(TemporaryTest):
    def build(self, chosen, rejected, future, *flags):
        for name, rows in [('chosen', chosen), ('rejected', rejected), ('future', future)]:
            (self.root / (name + '.json')).write_text(json.dumps(rows))
        out = self.root / 'pairs.jsonl'
        command = [sys.executable, str(ROOT / 'settings/tsr/reformat2DPO_future.py'),
                   '--input_path_chosen', str(self.root / 'chosen.json'),
                   '--input_path_rejected', str(self.root / 'rejected.json'),
                   '--input_path_future', str(self.root / 'future.json'), '--output_path', str(out), *flags]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            return result, []
        return result, [json.loads(line) for line in out.read_text().splitlines()]

    def test_default_future_tie_and_current_rejected_tie(self):
        current = [{'prompt': 'q', 'responses': ['current best', 'current worst'], 'scores': [5, 1]}]
        past = [{'prompt': 'q', 'responses': ['past worst'], 'scores': [1]}]
        future = [{'prompt': 'q', 'responses': ['future first', 'future second'], 'scores': [5, 5]}]
        result, rows = self.build(current, past, future)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['chosen'][-1]['content'], 'future first')
        self.assertEqual(rows[0]['rejected'][-1]['content'], 'current worst')
        result, strict = self.build(current, past, future, '--strict')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(strict[0]['chosen'][-1]['content'], 'current best')

    def test_filter_removes_only_nonpositive_score_gap(self):
        rows = [{'prompt': 'q', 'responses': ['first', 'second'], 'scores': [5, 5]}]
        result, kept = self.build(rows, rows, rows)
        self.assertEqual(len(kept), 1)
        result, removed = self.build(rows, rows, rows, '--filter')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(removed, [])

    def test_misaligned_prompts_fail_without_name_error(self):
        a = [{'prompt': 'q', 'responses': ['x'], 'scores': [5]}]
        b = [{'prompt': 'other', 'responses': ['x'], 'scores': [5]}]
        result, _ = self.build(a, b, a)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('NameError', result.stderr)


class TokenizationTests(unittest.TestCase):
    class Tokens:
        bos_token_id = 1
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            return {'input_ids': [int(x) for x in text.split()]}

    def tokenize(self, prompt, tokenizer=None, max_prompt_length=512):
        return TSRDPOTrainer.tokenize_row({'prompt': prompt, 'chosen': '7 8', 'rejected': '9'},
                                          tokenizer or self.Tokens(), max_prompt_length, None, False)

    def test_adds_one_bos_and_preserves_answer_eos(self):
        result = self.tokenize('4 5')
        self.assertEqual(result['prompt_input_ids'], [1, 4, 5])
        self.assertEqual(result['chosen_input_ids'], [7, 8, 2])
        self.assertEqual(result['rejected_input_ids'], [9, 2])
        self.assertEqual(self.tokenize('1 4')['prompt_input_ids'], [1, 4])
        self.assertEqual(self.tokenize('')['prompt_input_ids'], [1])

    def test_verified_post_truncation_behavior(self):
        result = self.tokenize(' '.join(str(x) for x in range(10, 610)))
        self.assertEqual(len(result['prompt_input_ids']), 513)
        self.assertEqual(result['prompt_input_ids'][1:], list(range(98, 610)))

    def test_missing_bos_and_original_trl_are_unchanged(self):
        tokenizer = self.Tokens()
        tokenizer.bos_token_id = None
        self.assertEqual(self.tokenize('4 5', tokenizer)['prompt_input_ids'], [4, 5])
        unpatched = DPOTrainer.tokenize_row({'prompt': '4', 'chosen': '7', 'rejected': '9'}, self.Tokens(), 512, None, False)
        self.assertEqual(unpatched['prompt_input_ids'], [4])


class LaunchTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.project = self.root / 'project'
        shutil.copytree(ROOT / 'scripts', self.project / 'scripts')
        for d in ['alignment-handbook', 'tsr', 'settings']:
            (self.project / d).mkdir()
        bin_dir = self.root / 'bin'; bin_dir.mkdir()
        stub = '#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["CALL_LOG"],"a") as f: f.write(json.dumps(sys.argv)+"\\n")\nif os.environ.get("FAIL_PYTHON") and sys.argv[0].endswith("/python"): sys.exit(7)\n'
        for name in ['python', 'accelerate']:
            p = bin_dir / name; p.write_text(stub); p.chmod(0o755)
        self.log = self.root / 'calls.jsonl'
        self.env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'], CALL_LOG=str(self.log),
                        CONDA_PATH='', HF_TOKEN='', CUDA_VISIBLE_DEVICES='0,1',
                        TSR_ITER1_STRICT_FUTURE='0', TSR_ITER1_FILTER_EQUAL_SCORES='0')

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_sft_zero3_global_batch_and_base_override(self):
        env = dict(self.env, SFT_BASE_MODEL='/models/base with spaces')
        subprocess.run(['bash', str(self.project / 'scripts/sft.sh')], env=env, check=True, capture_output=True)
        args = self.calls()[0]
        self.assertTrue(any(x.endswith('/settings/tsr/config/deepspeed_zero3.yaml') for x in args))
        self.assertIn('--per_device_train_batch_size=2', args)
        self.assertIn('--gradient_accumulation_steps=8', args)
        self.assertIn('--model_name_or_path=/models/base with spaces', args)

    def test_batch_32_for_supported_gpu_counts(self):
        for world in [2, 4, 8]:
            for mode, micro in [('sft', 2), ('dpo', 1)]:
                code = 'source "$1"; PROJECT_ROOT="$2"; WORLD_SIZE="$3"; launch_training "$4" config.yaml'
                subprocess.run(['bash', '-eu', '-c', code, 'test', str(self.project / 'scripts/common.sh'),
                                str(self.project), str(world), mode], env=self.env, check=True, capture_output=True)
                args = self.calls()[-1]
                self.assertIn(f'--per_device_train_batch_size={micro}', args)
                self.assertIn(f'--gradient_accumulation_steps={32//world//micro}', args)

    def test_filter_is_opt_in_and_only_applies_to_iter1_d2(self):
        for enabled in ['0', '1']:
            self.log.unlink(missing_ok=True)
            result = subprocess.run(['bash', str(self.project / 'scripts/tsr.sh')],
                                    env=dict(self.env, TSR_ITER1_FILTER_EQUAL_SCORES=enabled), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            future = [c for c in self.calls() if any(x.endswith('reformat2DPO_future.py') for x in c)]
            past = [c for c in self.calls() if any(x.endswith('reformat2DPO_past.py') for x in c)]
            self.assertEqual(len(future), 2)
            self.assertTrue(all('--filter' not in c for c in past))
            self.assertNotIn('--filter', future[0])
            self.assertEqual('--filter' in future[1], enabled == '1')

    def test_first_generation_failure_stops_training(self):
        result = subprocess.run(['bash', str(self.project / 'scripts/tsr.sh')],
                                env=dict(self.env, FAIL_PYTHON='1'), capture_output=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(self.calls()), 1)


class RealTrainerTests(TemporaryTest):
    def tiny_model(self):
        backend = Tokenizer(WordLevel({'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3,
                                      'question': 4, 'good': 5, 'bad': 6, 'answer': 7}, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', bos_token='[BOS]',
                                           eos_token='[EOS]', unk_token='[UNK]')
        model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=64, n_embd=16, n_layer=1, n_head=2,
                                         bos_token_id=1, eos_token_id=2, pad_token_id=0))
        return model, tokenizer

    def test_real_cpu_dpo_optimizer_step(self):
        torch.set_num_threads(2)
        model, tokenizer = self.tiny_model()
        dataset = Dataset.from_list([{'prompt': 'question', 'chosen': 'good answer', 'rejected': 'bad answer'}] * 2)
        config = DPOConfig(output_dir=str(self.root / 'dpo'), use_cpu=True, bf16=False, fp16=False,
                           max_steps=1, per_device_train_batch_size=1, learning_rate=1e-3,
                           max_length=32, max_prompt_length=16, report_to=[], save_strategy='no', disable_tqdm=True)
        before = model.transformer.wte.weight.detach().clone()
        trainer = TSRDPOTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer)
        self.assertEqual(trainer.train_dataset[0]['prompt_input_ids'][0], tokenizer.bos_token_id)
        trainer.train()
        self.assertEqual(trainer.state.global_step, 1)
        self.assertFalse(torch.equal(before, model.transformer.wte.weight.detach()))

    def test_real_cpu_sft_step_full_token_loss_and_model_card(self):
        model, tokenizer = self.tiny_model()
        config = SFTConfig(output_dir=str(self.root / 'sft'), use_cpu=True, max_seq_length=32,
                           dataset_text_field='text', report_to=[], max_steps=1,
                           per_device_train_batch_size=1, learning_rate=1e-3,
                           bf16=False, fp16=False, save_strategy='no', disable_tqdm=True)
        trainer = SFTTrainer(model=model, args=config, tokenizer=tokenizer,
                             train_dataset=Dataset.from_list([{'text': 'question good answer'}]))
        batch = trainer.data_collator([trainer.train_dataset[0]])
        self.assertTrue(torch.equal(batch['input_ids'], batch['labels']))
        before = model.transformer.wte.weight.detach().clone()
        trainer.train()
        self.assertEqual(trainer.state.global_step, 1)
        self.assertFalse(torch.equal(before, model.transformer.wte.weight.detach()))
        tree = ast.parse((ROOT / 'alignment-handbook/scripts/run_sft.py').read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'create_model_card')
        names = {k.arg for k in call.keywords}
        self.assertEqual(names, {'model_name', 'dataset_name', 'tags'})
        self.assertTrue(names <= set(inspect.signature(trainer.create_model_card).parameters))
        trainer.create_model_card(model_name='tiny-test', dataset_name='fixture', tags=['test'])
        self.assertTrue((self.root / 'sft/README.md').is_file())


if __name__ == '__main__':
    unittest.main()
