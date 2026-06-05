import numpy as np
import cv2
import pymunk


from typing import Optional
from dataclasses import dataclass

from ar_shuffleboard.utils.constants import GameConfig


@dataclass
class ScoreArea:
    score: int
    ratio: float
    color: tuple[int, int, int, int]  # BGRA


SCORE_AREA_LIST: list[ScoreArea] = [  # 점수 영역 데이터 리스트 (점수, 영역 비율)
    ScoreArea(3, 0.1, (0, 0, 255, 80)),  # red
    ScoreArea(2, 0.1, (0, 165, 255, 80)),  # orange
    ScoreArea(1, 0.1, (0, 255, 255, 80)),  # yellow
]


@dataclass
class Puck:
    body: pymunk.Body
    shape: pymunk.Shape


@dataclass
class PlayerData:
    name: str
    color: tuple[int, int, int, int]  # BGRA
    total_pucks: int
    left_pucks: int
    pucks: list[Puck]
    score: int

    def __init__(self, name: str, puck_count: int, color: tuple[int, int, int, int]):
        self.name = name
        self.total_pucks = puck_count
        self.left_pucks = puck_count
        self.pucks = []
        self.color = color
        self.score = 0

    def reset(self):
        self.pucks.clear()
        self.left_pucks = self.total_pucks
        self.score = 0


PLAYER_COLOR_LIST: list[tuple[int, int, int]] = [  # BGRA
    (255, 0, 0, 255),  # blue
    (0, 0, 255, 255),  # red
    (0, 255, 0, 255),  # green
    (0, 255, 255, 255),  # yellow
]


class Shuffleboard:
    def __init__(
        self,
        board_size: tuple[int, int],
    ):
        self.board_size = board_size  # 보드 크기 (w, h) (mm단위)
        self.puck_radius = GameConfig.PUCK_RADIUS  # 퍽의 반지름
        self.pucks_per_player = GameConfig.PUCKS_PER_PLAYER  # 플레이어 당 퍽 개수
        self.player_list = [PlayerData(f"P{i+1}", self.pucks_per_player, PLAYER_COLOR_LIST[i]) for i in range(np.clip(GameConfig.PLAYERS, 2, 4))]

        self.current_player_idx: int = 0  # 현재 플레이어 인덱스
        self.current_puck: Optional[Puck] = None

        # Pymunk
        self.space = pymunk.Space()
        self.space.damping = 0.65

    # === Game Manager ===

    def placePuck(self, position: Optional[tuple[int, int]] = None, player: Optional[PlayerData] = None):
        """게임 보드에 퍽을 배치한다."""
        if player is None:
            player = self.getCurrentPlayer()
        body = pymunk.Body(10, 500)
        if position is None:
            w, h = self.board_size
            position = (w, h // 2)
        body.position = position
        shape = pymunk.Circle(body, self.puck_radius)
        self.space.add(body, shape)

        self.current_puck = Puck(body, shape)
        player.pucks.append(self.current_puck)

    def hitPuck(self, puck: Puck, velocity: tuple[int, int]):
        """퍽의 속도를 직접 설정한다."""
        puck.body.velocity = velocity

    def hitPuckForReady(self, velocity: tuple[int, int]):
        """준비된 퍽을 친다."""
        if self.current_puck is None:
            return
        self.hitPuck(self.current_puck, velocity)

    def updateScores(self):
        """퍽의 위치를 기반으로 점수를 업데이트한다."""
        for player in self.player_list:
            player.score = 0
            for puck in player.pucks:
                player.score += self.getScoreByPosition(puck)

    def stepSpace(self, dt: int):
        """시뮬레이션 공간에서 시간의 흐름을 진행한다."""
        self.space.step(dt / 1000.0)

    # === Utilities ===

    def setPlayerConfig(self, id: int, name: Optional[str] = None, puck_color: Optional[tuple[int, int, int]] = None):
        """플레이어 정보(이름, 퍽 색상)을 설정한다."""
        player = self.player_list[id]
        if name is not None:
            player.name = name
        if puck_color is not None:
            player.puck_color = puck_color

    def setScore(self, id: int, score: int):
        """플레이어의 점수를 설정한다."""
        self.player_list[id].score = score

    def getScoreByPosition(self, puck: Puck) -> int:
        """위치에 따른 점수를 반환한다."""
        w, h = self.board_size
        x, y = puck.body.position
        # 좌우 모서리 벗어낫는지 확인
        if y + self.puck_radius <= 0 or y - self.puck_radius >= h:
            return 0
        # 점수 영역 내부인지 확인
        prev_ratio = 0.0
        for score_area in SCORE_AREA_LIST:
            next_ratio = prev_ratio + score_area.ratio
            if prev_ratio * w - self.puck_radius < x < next_ratio * w + self.puck_radius:
                return score_area.score
            prev_ratio = next_ratio

        return 0

    def setNextPlayer(self):
        """다음 플레이어로 설정한다."""
        self.current_player_idx += 1
        if self.current_player_idx == len(self.player_list):
            self.current_player_idx = 0

    # === Getter Methods ===

    def getPlacedPucks(self) -> list[Puck]:
        """배치된 퍽들을 반환한다."""
        return self.placed_puck_list

    def getCurrentPlayer(self) -> PlayerData:
        """현재 플레이어를 반환한다."""
        return self.player_list[self.current_player_idx]

    def getPlayers(self):
        """플레이어 리스트를 반환한다."""
        return self.player_list


""" 

0% 10% 20% 30%                100%
(0, 0) ------------------- (w, 0)
|   |   |   |                   |
|   |   |   |                   |
|   |   |   |                   |
| 3 | 2 | 1 |                   | <- start position
|   |   |   |                   |
|   |   |   |                   |
|   |   |   |                   |
(0, h) ------------------- (w, h)

"""
