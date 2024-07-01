import os
import logging
import warnings

from proteinchat.common.registry import registry
from proteinchat.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from proteinchat.datasets.datasets.seq_dataset import SeqDataset

@registry.register_builder("seq")
class SeqBuilder(BaseDatasetBuilder):
    train_dataset_cls = SeqDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/seq/seq.yaml",
    }
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")

        build_info = self.config.build_info
        datasets = dict()       
        
        # for split in ['train', 'valid']:
        for split in ['train']:
            storage_path = build_info.get(split).storage
            if not os.path.exists(storage_path):
                warnings.warn("storage path {} does not exist.".format(storage_path))

            is_train = split == "train"
            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                kw_path = os.path.join(storage_path, 'qa_kw.json'),
                text_rule_path = os.path.join(storage_path, 'qa_text_rule.json'),
                text_manual_path = os.path.join(storage_path, 'qa_text_manual.json'),
                seq_path = os.path.join(storage_path, 'seq.json'),
            )

        return datasets

