import random
import string
import sys
from enum import Enum

from loguru import logger

from velocyto import logic


class LogicType(str, Enum):
    Permissive10X = "Permissive10X"
    Intermediate10X = "Intermediate10X"
    ValidatedIntrons10X = "ValidatedIntrons10X"
    Stricter10X = "Stricter10X"
    ObservedSpanning10X = "ObservedSpanning10X"
    Discordant10X = "Discordant10X"
    SmartSeq2 = "SmartSeq2"


class UMIExtension(str, Enum):
    no = "no"
    char = "chr"
    gene = "Gene"
    Nbp = "[N]bp"


class LoomdType(str, Enum):
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"


def id_generator(size: int = 6, chars: str = string.ascii_uppercase + string.digits) -> str:
    return "".join(random.choice(chars) for _ in range(size))  # noqa: S311


def choose_logic(
    choice: LogicType,
) -> (
    logic.Permissive10X
    | logic.Intermediate10X
    | logic.ValidatedIntrons10X
    | logic.Stricter10X
    | logic.ObservedSpanning10X
    | logic.Discordant10X
    | logic.SmartSeq2
):
    match choice:
        case "Permissive10X":
            return logic.Permissive10X
        case "Intermediate10X":
            return logic.Intermediate10X
        case "ValidatedIntrons10X":
            return logic.ValidatedIntrons10X
        case "Stricter10X":
            return logic.Stricter10X
        case "ObservedSpanning10X":
            return logic.ObservedSpanning10X
        case "Discordant10X":
            return logic.Discordant10X
        case "SmartSeq2":
            return logic.SmartSeq2
        case _:
            logger.exception(f"{choice.value} is not a valid logic type")
            sys.exit()


def choose_dtype(choice: LoomdType) -> str:
    if choice == "uint16":
        return "uint16"
    elif choice == "uint32":
        return "uint32"
    else:
        return "uint64"
