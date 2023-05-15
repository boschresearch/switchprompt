""" Utility classes and functions related to SwitchPrompt (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import re
import json
import shutil

import numpy as np
np.random.seed(123)

# Set this to the place where you extracted the MNLI files
base_path = '/path/to/mnli/data/'

train_file = 'multinli_1.0_train.txt'
dev_matched = 'multinli_1.0_dev_matched.txt'
dev_mismatched = 'multinli_1.0_dev_mismatched.txt'


def create_dir(path, clean=True):
    if clean:
        shutil.rmtree(path, ignore_errors=True)  # Delete directory and its content
    os.makedirs(path)  # Recreate the directory path
    return path

def read_mnli_file(filename):
    instances_per_genre = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.read().splitlines()):
            if i > 0 and line.strip():
                gold_label, _, _, _, _, sentence1, sentence2, _, _, genre, _, _, _, _, _ = line.split('\t')
                if genre not in instances_per_genre:
                    instances_per_genre[genre] = []
                instances_per_genre[genre].append({
                    "lang": "en", 
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "gold": gold_label,
                    "genre": genre,
                })
    return instances_per_genre

def get_examples(instances, label, k, offset=0):
    examples = []
    for j, inst in enumerate(instances[offset:]):
        if len(examples) == k:
            return examples, j+offset
        if inst['gold'] == label:
            examples.append(inst)
    return examples, len(instances)


create_dir('data/')

n_shots = [64, 16, 4, 2]
labels = ['neutral', 'entailment', 'contradiction']

for split, splitfile in [('mnli_train', train_file),
                         # We dont need mismatched, as we dont study cross-domain transfer
                         #('mnli_dev_mismatched', dev_mismatched),
                         ('mnli_dev', dev_matched)]:
    instances_per_genre = read_mnli_file(base_path + splitfile)
    
    all_examples = {n: [] for n in n_shots}
    for genre, instances in instances_per_genre.items():
        np.random.shuffle(instances)
        
        for label in labels:
            max_offset = 0
            for shots in n_shots:
                
                # Create two different splits as new test file is given
                # Use train_1 for training, train_2 for development, dev_X for testing
                if split == 'mnli_train':
                    with open(f'data/{split}_1_{genre}_{shots}_shots.jsonl', 'a', encoding='utf-8') as fout:
                        examples_1, offset = get_examples(instances, label, shots)
                        max_offset = max(max_offset, offset) # is only triggered at 64 shots
                        for element in examples_1:
                            fout.write(json.dumps(element, ensure_ascii=False)+'\n')
                            
                    with open(f'data/{split}_2_{genre}_{shots}_shots.jsonl', 'a', encoding='utf-8') as fout:
                        examples_2, _ = get_examples(instances, label, shots, max_offset)
                        for element in examples_2:
                            fout.write(json.dumps(element, ensure_ascii=False)+'\n')
                            
                else:
                    with open(f'data/{split}_{genre}_{shots}_shots.jsonl', 'a', encoding='utf-8') as fout:
                        examples, _ = get_examples(instances, label, shots)
                        for element in examples:
                            fout.write(json.dumps(element, ensure_ascii=False)+'\n')