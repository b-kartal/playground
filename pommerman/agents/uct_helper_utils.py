import numpy as np
import queue

from .. import constants
from .. import utility
from collections import defaultdict

# Pommerman Domain Information to Simulate the Game

all_actions = [constants.Action.Stop, constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down, constants.Action.Bomb]
directions = [constants.Action.Stop, constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]
_directionsAsVector = {constants.Action.Up: (0, 1),
                       constants.Action.Down: (0, -1),
                       constants.Action.Left: (1, 0),
                       constants.Action.Right: (-1, 0),
                       constants.Action.Stop: (0, 0)}
_directionsAsList = _directionsAsVector.items()


def _filter_legal_actions(state):
    my_position = tuple(state['position'])
    board = np.array(state['board'])
    enemies = [constants.Item(e) for e in state['enemies']]
    ret = [constants.Action.Bomb]
    for direction in directions:
        position = utility.get_next_position(my_position, direction)
        if utility.position_on_board(board, position) and utility.position_is_passable(board, position, enemies):
            ret.append(direction)
    return ret

def _walls(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Rigid.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _flame(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret


def _flame_counter(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 2 # TODO hardcode 2 is flame life if change - then this must be changed as well
    return ret


def _wood(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Wood.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _enemies_positions(board, enemies):
    ret = []
    for e in enemies:
        locations = np.where(board == e.value)
        for r, c in zip(locations[0], locations[1]):
            ret.append((r, c))
    return ret


def _agents_positions(board):
    ret = {}
    agent_ids = [10,11,12,13]
    for e in agent_ids:
        locations = np.where(board == e)
        for r, c in zip(locations[0], locations[1]):
            ret[e-10] = (r,c)
    return ret



def convert_bombs(bomb_map):
    ret = []
    locations = np.where(bomb_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({'position': (r, c), 'blast_strength': int(bomb_map[(r, c)])})
    return ret

def convert_flames(flame_map):
    ret = []
    locations = np.where(flame_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({'position': (r, c)})
    return ret

def board_analyze(board):

    def rigid_neighbor_counter(position, board):
        ret = 0
        if position[0]-1 >= 0 and board[position[0]-1][position[1]] == constants.Item.Rigid.value: #  NORTH
            ret +=1
        if position[0]+1 < constants.BOARD_SIZE and board[position[0]+1][position[1]] == constants.Item.Rigid.value: #  SOUTH
            ret +=1
        if position[1] + 1 < constants.BOARD_SIZE and board[position[0]][position[1] + 1] == constants.Item.Rigid.value:  # EAST
            ret += 1
        if position[1] - 1 >= 0 and board[position[0]][position[1] - 1] == constants.Item.Rigid.value:  # WEST
            ret += 1

        return ret

    board_output = np.zeros(board.shape)

    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if board[i][j] != constants.Item.Rigid.value:
                board_output[i][j] = rigid_neighbor_counter([i,j], board)
            else:
                board_output[i][j] = 10

    for i in range(constants.BOARD_SIZE):
        board_output[0][i] += 1
        board_output[0][constants.BOARD_SIZE-1] += 1
        board_output[i][0] += 1
        board_output[constants.BOARD_SIZE - 1][i] += 1

    print(board_output)

def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
    assert (depth is not None)

    if exclude is None:
        exclude = [
            constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
        ]

    def out_of_range(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return depth is not None and abs(y2 - y1) + abs(x2 - x1) > depth

    items = defaultdict(list)
    dist = {}
    prev = {}
    Q = queue.PriorityQueue()

    mx, my = my_position
    for r in range(max(0, mx - depth), min(len(board), mx + depth)):
        for c in range(max(0, my - depth), min(len(board), my + depth)):
            position = (r, c)
            if any([
                    out_of_range(my_position, position),
                    utility.position_in_items(board, position, exclude),
            ]):
                continue



            if position == my_position:
                dist[position] = 0
            else:
                dist[position] = np.inf

            prev[position] = None
            Q.put((dist[position], position))

    for bomb in bombs:
        if bomb['position'] == my_position:
            items[constants.Item.Bomb].append(my_position)

    while not Q.empty():
        _, position = Q.get()

        if utility.position_is_passable(board, position, enemies):
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if new_position not in dist:
                    continue

                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position

        item = constants.Item(board[position])
        items[item].append(position)

    return items, dist, prev