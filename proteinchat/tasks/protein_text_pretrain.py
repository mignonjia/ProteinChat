"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from proteinchat.common.registry import registry
from proteinchat.tasks.base_task import BaseTask


@registry.register_task("protein_text_pretrain")
class ProteinTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    # def valid_step(self, model, samples):

        # run_cfg = slf.cfg.run_cfg
        # captions = model.generate(
        #     samples,
        #     use_nucleus_sampling=False,
        #     num_beams=self.num_beams,
        #     max_length=self.max_len,
        #     min_length=self.min_len,
        # )

        # img_ids = samples["image_id"]
        # for caption, img_id in zip(captions, img_ids):
        #     results.append({"caption": caption, "image_id": int(img_id)})

        # return results

    # def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        # metrics = [1]
        # eval_result_file = self.save_result(
        #     result=val_result,
        #     result_dir=registry.get_path("result_dir"),
        #     filename="{}_epoch{}".format(split_name, epoch),
        #     remove_duplicate="image_id",
        # )

        # if self.report_metric:
        #     metrics = self._report_metrics(
        #         eval_result_file=eval_result_file, split_name=split_name
        #     )
        # else:
        #     metrics = {"agg_metrics": 0.0}

        # return metrics


