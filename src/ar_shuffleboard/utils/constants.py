from pathlib import Path
from typing import Final


class OS:
    PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

    CONFIG_DIR: Final[Path] = PROJECT_ROOT / "config"
    USER_CONFIG_PATH: Final[Path] = CONFIG_DIR / "config.cfg"
    DEFAULT_CONFIG_PATH: Final[Path] = CONFIG_DIR / "config_default.cfg"


class Keycode:
    # 키코드 → 문자열 매핑 (특수키 전용)
    _CODE_TO_STR = {
        13: "ENTER",
        27: "ESC",
        32: "SPACE",
    }

    @staticmethod
    def to_str(keycode: int) -> str:
        # 알파벳(대소문자)
        if 65 <= keycode <= 90 or 97 <= keycode <= 122:
            return chr(keycode)
        # 숫자
        if 48 <= keycode <= 57:
            return chr(keycode)
        # 매핑된 특수키
        if keycode in Keycode._CODE_TO_STR:
            return Keycode._CODE_TO_STR[keycode]
        # 기타
        return f"KEYCODE_{keycode}"

    # Uppercase letters (ASCII)
    A: Final[int] = ord("A")
    B: Final[int] = ord("B")
    C: Final[int] = ord("C")
    D: Final[int] = ord("D")
    E: Final[int] = ord("E")
    F: Final[int] = ord("F")
    G: Final[int] = ord("G")
    H: Final[int] = ord("H")
    I: Final[int] = ord("I")
    J: Final[int] = ord("J")
    K: Final[int] = ord("K")
    L: Final[int] = ord("L")
    M: Final[int] = ord("M")
    N: Final[int] = ord("N")
    O: Final[int] = ord("O")
    P: Final[int] = ord("P")
    Q: Final[int] = ord("Q")
    R: Final[int] = ord("R")
    S: Final[int] = ord("S")
    T: Final[int] = ord("T")
    U: Final[int] = ord("U")
    V: Final[int] = ord("V")
    W: Final[int] = ord("W")
    X: Final[int] = ord("X")
    Y: Final[int] = ord("Y")
    Z: Final[int] = ord("Z")

    # Lowercase letters (ASCII)
    a: Final[int] = ord("a")
    b: Final[int] = ord("b")
    c: Final[int] = ord("c")
    d: Final[int] = ord("d")
    e: Final[int] = ord("e")
    f: Final[int] = ord("f")
    g: Final[int] = ord("g")
    h: Final[int] = ord("h")
    i: Final[int] = ord("i")
    j: Final[int] = ord("j")
    k: Final[int] = ord("k")
    l: Final[int] = ord("l")
    m: Final[int] = ord("m")
    n: Final[int] = ord("n")
    o: Final[int] = ord("o")
    p: Final[int] = ord("p")
    q: Final[int] = ord("q")
    r: Final[int] = ord("r")
    s: Final[int] = ord("s")
    t: Final[int] = ord("t")
    u: Final[int] = ord("u")
    v: Final[int] = ord("v")
    w: Final[int] = ord("w")
    x: Final[int] = ord("x")
    y: Final[int] = ord("y")
    z: Final[int] = ord("z")

    # Common control keys
    ESC: Final[int] = 27
    ENTER: Final[int] = 13
    SPACE: Final[int] = 32
