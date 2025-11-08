from .aime.aime_handler import AIMETaskHandler
from .irish_aime.irish_aime_handler import IrishAIMETaskHandler
from .amc23.amc23_handler import AMC23TaskHandler
from .apps.apps_handler import APPSTaskHandler
from .arc.arc_handler import ARCChallengeTaskHandler
from .base import TaskConfig, TaskHandler
from .gpqa_diamond.gpqa_diamond_handler import GPQADiamondTaskHandler
from .gsm8k.gsm8k_handler import GSM8KTaskHandler
from .livecodebench.livecodebench_handler import LiveCodeBenchTaskHandler
from .math.math_handler import MathTaskHandler
from .minervamath.minervamath_handler import MinervaMathTaskHandler
from .mmlu.mmlu_handler import MMLUProTaskHandler, MMLUTaskHandler
from .numina.numina_handler import NUMINATaskHandler
from .olympiadbench.olympiadbench_handler import OlympiadBenchMathTaskHandler
from .taco.taco_handler import TACOTaskHandler
from .lc2024.lc2024_handler import lc2024TaskHandler
from .aime_irish_thought.aime_irish_thought_handler import AIME_irish_thought_TaskHandler
from .irish_aime_irish_thought.irish_aime_irish_thought_handler import IrishAIME_irish_thought_TaskHandler
from .lc2024_irish_thought.lc2024_irish_thought_handler import lc2024_irish_thought_TaskHandler

TASK_HANDLER_MAP = {
    "numina": NUMINATaskHandler,
    "apps": APPSTaskHandler,
    "taco": TACOTaskHandler,
    "math": MathTaskHandler,
    "aime": AIMETaskHandler,
    "irish_aime": IrishAIMETaskHandler,
    "gpqa_diamond": GPQADiamondTaskHandler,
    "mmlu": MMLUTaskHandler,
    "mmlu_pro": MMLUProTaskHandler,
    "livecodebench": LiveCodeBenchTaskHandler,
    "gsm8k": GSM8KTaskHandler,
    "arc_c": ARCChallengeTaskHandler,
    "amc23": AMC23TaskHandler,
    "minervamath": MinervaMathTaskHandler,
    "olympiadbench_math": OlympiadBenchMathTaskHandler,
    "lc2024": lc2024TaskHandler,
    'aime_irish_thought': AIME_irish_thought_TaskHandler,
    'irish_aime_irish_thought': IrishAIME_irish_thought_TaskHandler,
    'lc2024_irish_thought': lc2024_irish_thought_TaskHandler
}

__all__ = [
    lc2024TaskHandler,
    AIMETaskHandler,
    IrishAIMETaskHandler,
    APPSTaskHandler,
    TACOTaskHandler,
    MathTaskHandler,
    AMC23TaskHandler,
    NUMINATaskHandler,
    GPQADiamondTaskHandler,
    MMLUTaskHandler,
    MMLUProTaskHandler,
    LiveCodeBenchTaskHandler,
    GSM8KTaskHandler,
    ARCChallengeTaskHandler,
    TaskHandler,
    MathTaskHandler,
    OlympiadBenchMathTaskHandler,
    MinervaMathTaskHandler,
    TaskConfig,
    TASK_HANDLER_MAP,
]
