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

# Lint as: python3
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import datasets
import json


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{lange2022clin,
  title={CLIN-X: pre-trained language models and a study on cross-task transfer for concept extraction in the clinical domain},
  author={Lange, Lukas and Adel, Heike and Str{\"o}tgen, Jannik and Klakow, Dietrich},
  journal={Bioinformatics},
  volume={38},
  number={12},
  pages={3267--3274},
  year={2022},
  publisher={Oxford University Press}
}
"""

_DESCRIPTION = """\
This is a question answer classification dataset
"""

_URL = "../../../clinic_dataset/"
_TRAINING_FILE = "train_split,2.jsonl"
_DEV_FILE = "dev_split,2.jsonl"
_TEST_FILE = "test_split,2.jsonl"


class clinicConfig(datasets.BuilderConfig):
    """BuilderConfig for CLIN-X"""

    def __init__(self, **kwargs):
        """BuilderConfig for CLIN-X.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(clinicConfig, self).__init__(**kwargs)


class clinic(datasets.GeneratorBasedBuilder):
    """CLIN-X dataset."""

    BUILDER_CONFIGS = [
        clinicConfig(name="clinic", version=datasets.Version("1.0.0"), description="Clinic Dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence_1": datasets.Value("string"),
                    "label": 
                        datasets.features.ClassLabel(
                            names=    ["positive", "negative"] #["Management", "Information", "Susceptibility", "Prognosis", "Diagnosis", "OtherEffect", "Cause", "Manifestation", "PersonOrg", "Complication", "Anatomy", "NotDisease"]
                        ),
                }
            ),
            supervised_keys=None,
            homepage=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            # sentence_1 = []
            # gold = []
            count = 0
            for line in f:
                row = json.loads(line)
                # if (count < 10):
                guid += 1
                sentence_1 = row.get("sentence1")
                gold = row.get("gold")
                
                yield guid, {
                                "id": str(guid),
                                "sentence_1": sentence_1,
                                "label": gold,
                            }
                
                    # count = count + 1