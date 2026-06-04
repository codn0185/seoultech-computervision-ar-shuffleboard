import numpy as np
import cv2
import pymunk


from typing import Optional
from dataclasses import dataclass


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
        puck_radius: int,
        players: int = 2,
        pucks_per_player: int = 4,
    ):
        self.board_size = board_size  # 보드 크기 (w, h) (mm단위)
        self.puck_radius = puck_radius  # 퍽의 반지름
        # self.player_list = [Player(i, f"P{i+1}", 0, PUCK_COLORS[i]) for i in range(np.clip(players, 2, 4))]  # 플레이어 정보
        self.pucks_per_player = pucks_per_player  # 플레이어 당 퍽 개수
        self.player_list = [PlayerData(f"P{i+1}", pucks_per_player, PLAYER_COLOR_LIST[i]) for i in range(np.clip(players, 2, 4))]

        self.current_player_idx: int = 0  # 현재 플레이어 인덱스
        self.current_puck: Optional[Puck] = None

        # Pymunk
        self.space = pymunk.Space()
        self.space.damping = 0.65

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

    def updateScores(self):
        """퍽의 위치를 기반으로 점수를 업데이트한다."""
        for player in self.player_list:
            player.score = 0
            for puck in player.pucks:
                player.score += self.getScoreByPosition(puck)

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

    def getPlacedPucks(self) -> list[Puck]:
        """배치된 퍽들을 반환한다."""
        return self.placed_puck_list

    def stepSpace(self, dt: int):
        """시뮬레이션 공간에서 시간의 흐름을 진행한다."""
        self.space.step(dt / 1000.0)

    def getCurrentPlayer(self) -> PlayerData:
        """현재 플레이어를 반환한다."""
        return self.player_list[self.current_player_idx]

    def setNextPlayer(self):
        """다음 플레이어로 설정한다."""
        self.current_player_idx += 1
        if self.current_player_idx == len(self.player_list):
            self.current_player_idx = 0

    # === Display Methods ===

    def getGameImage(self) -> np.ndarray:
        """게임 이미지를 반환한다. (보드 사이즈의 2배)"""
        w, h = self.board_size
        img = np.zeros((2 * h, 2 * w, 4), dtype=np.uint8)
        origin = self.getGameOrigin()
        self.drawScoreArea(img, origin)
        self.drawBoarderLines(img, origin)
        self.drawPucks(img, origin)
        return img

    def getGameOrigin(self) -> tuple[int, int]:
        """게임 이미지 내 게임 보드의 오프셋을 반환한다."""
        w, h = self.board_size
        return (w // 2, h // 2)

    def drawScoreArea(self, img: np.ndarray, offset: tuple[int, int]):
        """점수 영역 내부를 색칠한다."""
        x, y = offset
        w, h = self.board_size

        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            x1 = int(x + (prev_area - score_area.ratio) * w)
            x2 = int(x + prev_area * w)
            color = score_area.color
            cv2.rectangle(img, (x1, y), (x2, y + h), color, -1)

    def drawBoarderLines(self, img: np.ndarray, offset: tuple[int, int]):
        """게임 보드에 경계선을 그린다."""
        x, y = offset
        w, h = self.board_size

        line_color = (238, 238, 238, 255)
        # 모서리 경계선
        cv2.line(img, (x, y), (x + w, y), line_color, 2)  # 위
        cv2.line(img, (x, y + h), (x + w, y + h), line_color, 2)  # 아래
        cv2.line(img, (x, y), (x, y + h), line_color, 2)  # 좌
        cv2.line(img, (x + w, y), (x + w, y + h), line_color, 2)  # 우
        # 점수 경계선
        prev_area = 0.0
        for score_area in SCORE_AREA_LIST:
            prev_area += score_area.ratio
            _x = int(x + prev_area * w)
            cv2.line(img, (_x, y), (_x, y + h), line_color, 2)

    def drawPucks(self, img: np.ndarray, offset: tuple[int, int]):
        """게임 보드에 퍽을 그린다."""
        x, y = offset

        for player in self.player_list:
            for puck in player.pucks:
                cx, cy = puck.body.position
                center = (int(cx) + x, int(cy) + y)
                cv2.circle(img, center, self.puck_radius, player.color, -1)


def main():
    fps = 12
    w, h = (25 * 12, 25 * 5)
    game = Shuffleboard(board_size=(w, h), puck_radius=10)

    dt = 1000 // fps
    while True:
        game.stepSpace(dt)
        img = game.getGameImage()
        cv2.imshow("Shuffleboard Test", img)
        keycode = cv2.waitKey(dt)

        match keycode:
            case 27:  # ESC
                break
            case 32:  # SPACE
                print("SPACE 키 입력")
                game.placePuck((w, h // 2))
                game.setNextPlayer()
            case 13:  # ENTER
                game.hitPuckForReady((-200, 0))
            case 49:  #
                game.hitPuckForReady((-10, 0))
            case 50:  #
                game.hitPuckForReady((-20, 0))
            case 51:  #
                game.hitPuckForReady((-30, 0))
            case 52:  #
                game.hitPuckForReady((-40, 0))
            case 53:  #
                game.hitPuckForReady((-50, 0))


if __name__ == "__main__":
    main()

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
