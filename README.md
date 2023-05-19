<!---

    Copyright (c) 2022 Robert Bosch GmbH and its subsidiaries.

-->

# SwitchPrompt

This repository contains the companion material for the following publication:

> Koustava Goswami, Lukas Lange, Jun Araki, Heike Adel. SwitchPrompt: Learning Domain-Specific Gated Soft Prompts for Classification in Low-Resource Domains. EACL 2023.

Please cite this paper if using the code or references. The paper can be found at [ArXix](https://arxiv.org/abs/2302.06868) and [ACL Anthology](https://aclanthology.org/2023.eacl-main.197/).

```
@inproceedings{goswami-etal-2023-switchprompt,
    title = "{S}witch{P}rompt: Learning Domain-Specific Gated Soft Prompts for Classification in Low-Resource Domains",
    author = "Goswami, Koustava  and
      Lange, Lukas  and
      Araki, Jun  and
      Adel, Heike",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.197",
    pages = "2689--2695",
}
```

## Purpose of this Software

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## Description

Train your own models

To create your own few-shot datasets, you need to run the following script 
```
python databuilding_script.py
```

The components that are required to be replaced are:-

1. The train, test path needs to be mentioned in the variable train_file and dev_matched respectively. The development file will be created from train dataset. The dev_matched will create the test dataset.
2. The number of few shots will be mentioned in the variable n_shots
3. The labels of the dataset will be stored in the variable labels
4. The controller dataset code will come under tasks folder


Dataset_Run Controller

Folder Path: tasks/clinic/datasets/clinic.py [Note: For new dataset the $task$.py needs to be placed here]
```
tasks_URL = $task_name_dataset_location$
_TRAINING_FILE = $train_file_name$
_DEV_FILE = $dev_file_name$
_TEST_FILE = $test_file_name$
```

Train model on Clinical dataset

```
python run.py \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--output_dir models/clinic_bert_n_shots \
--overwrite_output_dir \
--hidden_dropout_prob 0.1 \
--seed 11 \
--save_strategy no \
--evaluation_strategy epoch \
--prefix \
--model_name_or_path bert-base-cased \
--task_name clinic \
--dataset_name clinic \
--num_static_keyword 10
--num_dynamic_keyword 0
--pre_seq_len 6
--generic_dataset $generic_dataset_path$
--specific_dataset $task_dataset_path$
--clinic
```

Train model on SOFC dataset
```
python run.py \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--output_dir models/sofc_bert_n_shots \
--overwrite_output_dir \
--hidden_dropout_prob 0.1 \
--seed 11 \
--save_strategy no \
--evaluation_strategy epoch \
--prefix \
--model_name_or_path bert-base-cased \
--task_name sofc \
--dataset_name sofc \
--num_static_keyword 10
--num_dynamic_keyword 0
--pre_seq_len 6
--generic_dataset $generic_dataset_path$
--specific_dataset $task_dataset_path$
--other
```

Train model on TREC dataset
```
python run.py \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--output_dir models/trec_bert_n_shots \
--overwrite_output_dir \
--hidden_dropout_prob 0.1 \
--seed 11 \
--save_strategy no \
--evaluation_strategy epoch \
--prefix \
--model_name_or_path bert-base-cased \
--task_name trec \
--dataset_name trec \
--num_static_keyword 10
--num_dynamic_keyword 0
--pre_seq_len 6
--generic_dataset $generic_dataset_path$
--specific_dataset $task_dataset_path$
--other
```

For the keyword selection, it is prefferable to keep generic_dataset and specific_dataset in the same format but that can be controlled from the util.py in model folder. New model and code sequence controller can be added or modified from the utils.py code from the model folder.




## License

The code in this repository is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE.txt) file for details.
For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
