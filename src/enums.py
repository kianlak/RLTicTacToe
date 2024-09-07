from enum import Enum

class CellState(Enum):
  EMPTY = 0
  O = 1
  X = 2

class GameState(Enum):
  PLAYING = 0
  AGENT_WIN = 1
  PLAYER_WIN = 2
  TIE = 3