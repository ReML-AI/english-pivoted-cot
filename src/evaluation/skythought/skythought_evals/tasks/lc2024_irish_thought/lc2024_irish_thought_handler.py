from typing import Dict

from skythought_evals.util.model_utils import MODEL_TO_NAME

from ..math.math_handler import MathTaskHandler


class lc2024_irish_thought_TaskHandler(MathTaskHandler):
    def generate_prompt(self, problem: Dict, model):
        if MODEL_TO_NAME[model] == "Sky-T1-32B-Preview":
            return self.task_config.templating_parameters["sky_template"].format(
                prompt=problem["problem"]
            )
        else:
            return self.task_config.templating_parameters["regular_template"].format(
                prompt=problem["problem"]
            )

    def make_conversations(self, data, system_prompt, model=None):
        conversations = []
        for problem in data:
            prompt_text = self.generate_prompt(problem, model)
            conversations.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                    {
                        "role": "assistant",
                        "content": "<think>\n\nLet's think in Irish. ",
                    },
                ]
            )
        return conversations

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        train_data = self.load_dataset(subset=subset, split=split).to_pandas()
        return train_data
