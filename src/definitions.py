"""Global token and step-field definitions for the project.

Step-field name → token type id mapping is used by StreamStore and
OfflineDQNTransformer; pass FIELD_TO_TYPE into both.
"""

from enum import IntEnum


class TokenType(IntEnum):
    ACTION = 0
    OBS = 1
    REWARD = 2
    DONE = 3

# Step-field name → token type id. Required in StreamStore and OfflineDQNTransformer __init__.
# Key order is canonical token order: action, obs, reward, done.
FIELD_TO_TYPE: dict[str, int] = {
    "action": TokenType.ACTION,
    "observation": TokenType.OBS,
    "reward": TokenType.REWARD,
    "done": TokenType.DONE,
}