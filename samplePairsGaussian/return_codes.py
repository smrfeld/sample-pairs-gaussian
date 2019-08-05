from enum import Enum

class ReturnCode(Enum):
    SUCCESS = 1
    FAIL_ZERO_PARTICLES = 2
    FAIL_ONE_PARTICLE = 3
    FAIL_STD_CLIP_MULT = 4
    FAIL_NO_SAMPLING_TRIES = 5
