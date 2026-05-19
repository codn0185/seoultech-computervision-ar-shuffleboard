import configparser
import shutil

from .constants import USER_CONFIG_PATH, DEFAULT_CONFIG_PATH

__all__ = ["load_config"]


def load_config() -> dict:
    config = configparser.ConfigParser()

    # default config 확인 및 불러오기
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"기본 설정 파일이 없습니다. {DEFAULT_CONFIG_PATH}")
    config.read(DEFAULT_CONFIG_PATH, encoding="utf-8")

    # user config가 없으면 default config 복사
    if not USER_CONFIG_PATH.exists():
        shutil.copy(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
    else:
        # user config 파일 불러오기
        try:
            config.read(USER_CONFIG_PATH, encoding="utf-8")
        except:
            print(f"config.cfg 파일이 손상되어 기본 설정을 유지합니다.")

    return {section: dict(config.items(section)) for section in config.sections()}
