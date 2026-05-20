import configparser
import shutil

from .constants import OS

__all__ = ["load_config"]


def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()

    # default config 확인 및 불러오기
    if not OS.DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"기본 설정 파일이 없습니다. {OS.DEFAULT_CONFIG_PATH}")
    config.read(OS.DEFAULT_CONFIG_PATH, encoding="utf-8")

    # user config가 없으면 default config 복사
    if not OS.USER_CONFIG_PATH.exists():
        shutil.copy(OS.DEFAULT_CONFIG_PATH, OS.USER_CONFIG_PATH)
    else:
        # user config 파일 불러오기
        try:
            config.read(OS.USER_CONFIG_PATH, encoding="utf-8")
        except:
            print(f"config.cfg 파일이 손상되어 기본 설정을 유지합니다.")

    return config
    # return {section: dict(config.items(section)) for section in config.sections()}
