import numpy as np
import queue

from pommerman import constants
from pommerman import utility
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

def _passage(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Passage.value)
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

def _bombs(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Bomb.value)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _powerup_ExtraBomb(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.ExtraBomb)
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _powerup_IncRange(board):
    ret = np.zeros(board.shape)
    locations = np.where( board == constants.Item.IncrRange )
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret

def _powerup_Kick(board):
    ret = np.zeros(board.shape)
    locations = np.where(board == constants.Item.Kick )
    for r, c in zip(locations[0], locations[1]):
        ret[r][c] = 1
    return ret


def is_valid_position(board, position, direction, step):
    row, col = position
    invalid_values = [item.value for item \
                                        in [constants.Item.Rigid]]
    if utility.position_on_board(board,position)== False:
        return False

    if constants.Action(direction) == constants.Action.Stop:
        return True

    if constants.Action(direction) == constants.Action.Up:
        return row - step >= 0 and board[row-step][col] not in invalid_values

    if constants.Action(direction) == constants.Action.Down:
        return row + step < len(board) and board[row+step][col] not in invalid_values

    if constants.Action(direction) == constants.Action.Left:
        return col - step >= 0 and board[row][col-step] not in invalid_values

    if constants.Action(direction) == constants.Action.Right:
        return col + step < len(board[0]) and \
            board[row][col+step] not in invalid_values

    raise constants.InvalidAction("We did not receive a valid direction: ", direction)

def get_next_position_steps(position, direction, steps):
    x, y = position
    if direction == constants.Action.Right:
        return (x, y + steps)
    elif direction == constants.Action.Left:
        return (x, y - steps)
    elif direction == constants.Action.Down:
        return (x + steps, y)
    elif direction == constants.Action.Up:
        return (x - steps, y)
    return (x, y)

def _wood_positions(board):
    ret = []
    locations = np.where(board == constants.Item.Wood.value)
    for r, c in zip(locations[0], locations[1]):
        ret.append((r,c))
    return ret

def _flames_positions(board):
    ret = []
    locations = np.where(board == constants.Item.Flames.value)
    for r, c in zip(locations[0], locations[1]):
        ret.append((r, c))
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


    # agent alive or not?

    # for flames
    # binary - flame locations
    # integer - flame time remaining

def diagonalAgentId(ourAgentId):
    list = [constants.Item.Agent0.value ,constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value] # agent IDs from the list
    return list[(list.index(ourAgentId)+2)%len(list)]


def nonDiagonalAgents(ourAgentId):
    # returns two adjacent (non-diagonal) enemy ids
    list = [constants.Item.Agent0.value, constants.Item.Agent1.value, constants.Item.Agent2.value, constants.Item.Agent3.value]  # agent IDs from the list
    # get diagonal agent - remove diagonal and itself , return the other remaining two.
    list.remove(diagonalAgentId(ourAgentId))
    list.remove(ourAgentId)
    return list

def _teammate_position(board,e):
    ret = []
    locations = np.where(board == e.value)
    for r, c in zip(locations[0], locations[1]):
        ret.append((r, c))
    return ret

def generate_NN_input_with_ids(my_game_id ,observation, time_step): # latest channel version where one additional feature map with agent type
    #print(observation)

    numberOfChannels = 19
    current_board = np.array(observation['board'])  # done

    board_size = current_board.shape[0] # TODO ASSUMING THE BOARD IS SQUARE

    ret = np.zeros((numberOfChannels, board_size, board_size))  # inspired from backplay pommerman paper

    index_bomb_strength = 0  # integer
    index_bomb_t_explode = 1  # integer

    index_agent_loc = 2
    index_agent_bomb_count = 3
    index_agent_blast = 4
    index_agent_can_kick = 5
    index_agent_has_teammate = 6

    index_agent_mate_or_enemy1_loc = 7
    index_agent_enemy2_loc = 8
    index_agent_enemy3_loc = 9

    index_passage_loc = 10
    index_wall_loc = 11       # binary
    index_wood_loc = 12       # binary
    index_flame_loc = 13      # binary

    index_powerup_extrabomb = 14
    index_powerup_increaseblast = 15
    index_powerup_cankick = 16
    index_game_timestep = 17 # float % of game episode

    index_agentid_class = 18 # This will be 1 if agent id is 10 or 11, and 2 if agent id 12 or 13 ...


    #print("current board is \n")
    #print(current_board, "\n")

    alives = np.array(observation['alive'])

    our_agent_pos = np.array(observation['position'])
    #print("our position is ", our_agent_pos)
    our_agentid = my_game_id
    #print("our agent id is ", our_agentid)

    agent_positions = _agents_positions(current_board) # returns all the agent locations

   # print("agent positions are", agent_positions, " \n")

    enemies = np.array(observation['enemies'])

   # print("enemies are ", enemies)



   # print("alive agents are ", alives)

    ret[index_bomb_strength, :, :] = np.array(observation['bomb_blast_strength'])  # done

    #print( " bomb blast strength is ", ret[index_bomb_strength, :, :] )

    ret[index_bomb_t_explode, :, :] = np.array(observation['bomb_life'])  # done

    #print (" bomb time to explode is ", ret[index_bomb_t_explode,:,:])

    # print(f"agent locations {agent_positions} and our aget id {our_agentid}")

    if our_agentid in alives: # otherwise return our agent location empty or noT? TODO
        ret[index_agent_loc,agent_positions[our_agentid-10][0], agent_positions[our_agentid-10][1]] = 1 # binary map for our agent

    #print(" channel for our agent location ",ret[index_agent_loc,:,:])

    ret[index_agent_bomb_count,:,:] = np.ones((board_size,board_size)) * np.array(observation['ammo']) # integer full map

    #print(" channel for our agent ammo size ",ret[index_agent_bomb_count,:,:])

    ret[index_agent_blast,:,:] = np.ones((board_size, board_size)) * np.array(observation['blast_strength']) # integer map full

    #print(" channel for agent blast strength ",ret[index_agent_blast,:,:])

    ret[index_agent_can_kick,:,:] = np.ones((board_size, board_size)) * np.array(observation['can_kick']) # binary map full

    #print(" channel for agent kick ",ret[index_agent_can_kick,:,:])

    diagonalPlayer = diagonalAgentId(our_agentid) # TODO remove lines setting to zeros after testing

    if diagonalPlayer in alives and agent_positions.get(diagonalPlayer - 10,False):  #Fixed for partially pbservable version
        ret[index_agent_mate_or_enemy1_loc, agent_positions[diagonalPlayer - 10][0], agent_positions[diagonalPlayer - 10][1]] = 1  # set location to binary
        if np.array(observation['teammate']) != constants.Item.AgentDummy: # team game
            ret[index_agent_has_teammate,:,:] = np.ones((board_size, board_size)) # full binary map

    #print("team mate channel is ", ret[index_agent_has_teammate,:,:])
    #print("team mate location channel is ", ret[index_agent_mate_or_enemy1_loc, :, :])

    otherTwoEnemyIds = nonDiagonalAgents(our_agentid) # returns an array with two elements

  #  print("other agents are ", otherTwoEnemyIds)


    # first layer is the smallest id among two
    # second layer is the higher one

    if otherTwoEnemyIds[0] in alives and agent_positions.get(otherTwoEnemyIds[0]-10,False): #Fixed for partially pbservable version
        ret[index_agent_enemy2_loc,agent_positions[otherTwoEnemyIds[0]-10][0], agent_positions[otherTwoEnemyIds[0]-10][1]] = 1 # set location to binary

    if otherTwoEnemyIds[1] in alives and agent_positions.get(otherTwoEnemyIds[1]-10,False): #Fixed for partially pbservable version
        ret[index_agent_enemy3_loc, agent_positions[otherTwoEnemyIds[1]-10][0], agent_positions[otherTwoEnemyIds[1]-10][1]] = 1  # set location to binary

  #  print("enemy2 location channel is ", ret[index_agent_enemy2_loc, :, :])
  #  print("enemy3 location channel is ", ret[index_agent_enemy3_loc, :, :])


    ret[index_passage_loc,:,:] = _passage(current_board) # done
    ret[index_wall_loc,:,:] = _walls(current_board) # done
    ret[index_wood_loc,:,:] = _wood(current_board)  # done
    ret[index_flame_loc,:,:] = _flame(current_board) # done


    #print("passage is ", ret[index_passage_loc,:,:])
    #print("wall is ", ret[index_wall_loc,:,:])
    #print("wood is ", ret[index_wood_loc,:,:])
    #print("flame is ", ret[index_flame_loc,:,:])

    ret[index_powerup_extrabomb,:,:] = _powerup_ExtraBomb(current_board) # done
    ret[index_powerup_increaseblast, :, :] = _powerup_IncRange(current_board) # done
    ret[index_powerup_cankick, :, :] = _powerup_Kick(current_board) # done

    ignoreTimeIndex=False                    #This will partition the timesteps only in two phases
    if ignoreTimeIndex:
        if index_game_timestep<=40:
            ret[index_game_timestep, :, :] = np.zeros((board_size, board_size))
        else:
            ret[index_game_timestep, :, :] = np.ones((board_size, board_size))
    else:
        ret[index_game_timestep,:,:] = np.ones((board_size,board_size))*(time_step/800.0) # done TODO 800 game length here

    #print("time is ", ret[index_game_timestep,:,:])

    #print("generated channels")
    if our_agentid in (10,11):
        ret[index_agentid_class,:,:] = np.ones((board_size, board_size))
    else: # agent id is 12 or 13
        ret[index_agentid_class, :, :] = np.ones((board_size, board_size))*2


    #print(ret)

    return ret

    #print(" >>>>>>>>>>>>>>>>>>>>>>>>> \n")


def generate_NN_input(my_game_id ,observation, time_step):
    #print(observation)

    numberOfChannels = 18
    current_board = np.array(observation['board'])  # done

    board_size = current_board.shape[0] # TODO ASSUMING THE BOARD IS SQUARE

    ret = np.zeros((numberOfChannels, board_size, board_size))  # inspired from backplay pommerman paper

    index_bomb_strength = 0  # integer
    index_bomb_t_explode = 1  # integer

    index_agent_loc = 2
    index_agent_bomb_count = 3
    index_agent_blast = 4
    index_agent_can_kick = 5
    index_agent_has_teammate = 6

    index_agent_mate_or_enemy1_loc = 7
    index_agent_enemy2_loc = 8
    index_agent_enemy3_loc = 9

    index_passage_loc = 10
    index_wall_loc = 11       # binary
    index_wood_loc = 12       # binary
    index_flame_loc = 13      # binary

    index_powerup_extrabomb = 14
    index_powerup_increaseblast = 15
    index_powerup_cankick = 16
    index_game_timestep = 17 # float % of game episode

    #print("current board is \n")
    #print(current_board, "\n")

    alives = np.array(observation['alive'])

    our_agent_pos = np.array(observation['position'])
    #print("our position is ", our_agent_pos)
    our_agentid = my_game_id
    #print("our agent id is ", our_agentid)

    agent_positions = _agents_positions(current_board) # returns all the agent locations

   # print("agent positions are", agent_positions, " \n")

    enemies = np.array(observation['enemies'])

   # print("enemies are ", enemies)



   # print("alive agents are ", alives)

    ret[index_bomb_strength, :, :] = np.array(observation['bomb_blast_strength'])  # done

    #print( " bomb blast strength is ", ret[index_bomb_strength, :, :] )

    ret[index_bomb_t_explode, :, :] = np.array(observation['bomb_life'])  # done

    #print (" bomb time to explode is ", ret[index_bomb_t_explode,:,:])

    # print(f"agent locations {agent_positions} and our aget id {our_agentid}")

    if our_agentid in alives: # otherwise return our agent location empty or noT? TODO
        ret[index_agent_loc,agent_positions[our_agentid-10][0], agent_positions[our_agentid-10][1]] = 1 # binary map for our agent

    #print(" channel for our agent location ",ret[index_agent_loc,:,:])

    ret[index_agent_bomb_count,:,:] = np.ones((board_size,board_size)) * np.array(observation['ammo']) # integer full map

    #print(" channel for our agent ammo size ",ret[index_agent_bomb_count,:,:])

    ret[index_agent_blast,:,:] = np.ones((board_size, board_size)) * np.array(observation['blast_strength']) # integer map full

    #print(" channel for agent blast strength ",ret[index_agent_blast,:,:])

    ret[index_agent_can_kick,:,:] = np.ones((board_size, board_size)) * np.array(observation['can_kick']) # binary map full

    #print(" channel for agent kick ",ret[index_agent_can_kick,:,:])

    diagonalPlayer = diagonalAgentId(our_agentid) # TODO remove lines setting to zeros after testing

    if diagonalPlayer in alives and agent_positions.get(diagonalPlayer - 10,False):  #Fixed for partially pbservable version
        ret[index_agent_mate_or_enemy1_loc, agent_positions[diagonalPlayer - 10][0], agent_positions[diagonalPlayer - 10][1]] = 1  # set location to binary
        if np.array(observation['teammate']) != constants.Item.AgentDummy: # team game
            ret[index_agent_has_teammate,:,:] = np.ones((board_size, board_size)) # full binary map

    #print("team mate channel is ", ret[index_agent_has_teammate,:,:])
    #print("team mate location channel is ", ret[index_agent_mate_or_enemy1_loc, :, :])

    otherTwoEnemyIds = nonDiagonalAgents(our_agentid) # returns an array with two elements

  #  print("other agents are ", otherTwoEnemyIds)


    # first layer is the smallest id among two
    # second layer is the higher one

    if otherTwoEnemyIds[0] in alives and agent_positions.get(otherTwoEnemyIds[0]-10,False): #Fixed for partially pbservable version
        ret[index_agent_enemy2_loc,agent_positions[otherTwoEnemyIds[0]-10][0], agent_positions[otherTwoEnemyIds[0]-10][1]] = 1 # set location to binary

    if otherTwoEnemyIds[1] in alives and agent_positions.get(otherTwoEnemyIds[1]-10,False): #Fixed for partially pbservable version
        ret[index_agent_enemy3_loc, agent_positions[otherTwoEnemyIds[1]-10][0], agent_positions[otherTwoEnemyIds[1]-10][1]] = 1  # set location to binary

  #  print("enemy2 location channel is ", ret[index_agent_enemy2_loc, :, :])
  #  print("enemy3 location channel is ", ret[index_agent_enemy3_loc, :, :])


    ret[index_passage_loc,:,:] = _passage(current_board) # done
    ret[index_wall_loc,:,:] = _walls(current_board) # done
    ret[index_wood_loc,:,:] = _wood(current_board)  # done
    ret[index_flame_loc,:,:] = _flame(current_board) # done


    #print("passage is ", ret[index_passage_loc,:,:])
    #print("wall is ", ret[index_wall_loc,:,:])
    #print("wood is ", ret[index_wood_loc,:,:])
    #print("flame is ", ret[index_flame_loc,:,:])

    ret[index_powerup_extrabomb,:,:] = _powerup_ExtraBomb(current_board) # done
    ret[index_powerup_increaseblast, :, :] = _powerup_IncRange(current_board) # done
    ret[index_powerup_cankick, :, :] = _powerup_Kick(current_board) # done

    ignoreTimeIndex=False                    #This will partition the timesteps only in two phases
    if ignoreTimeIndex:
        if index_game_timestep<=40:
            ret[index_game_timestep, :, :] = np.zeros((board_size, board_size))
        else:
            ret[index_game_timestep, :, :] = np.ones((board_size, board_size))
    else:
        ret[index_game_timestep,:,:] = np.ones((board_size,board_size))*(time_step/800.0) # done TODO 800 game length here

    #print("time is ", ret[index_game_timestep,:,:])

    #print("generated channels")
    return ret

    #print(" >>>>>>>>>>>>>>>>>>>>>>>>> \n")

def generate_NN_input_team(my_game_id ,observation, my_game_id2, observation2,time_step):
    #print(observation)

    numberOfChannels = 20
    current_board = np.array(observation['board'])  # done

    board_size = current_board.shape[0] # TODO ASSUMING THE BOARD IS SQUARE

    ret = np.zeros((numberOfChannels, board_size, board_size))  # inspired from backplay pommerman paper

    index_bomb_strength = 0  # integer
    index_bomb_t_explode = 1  # integer

    index_agent_loc = 2
    index_agent_bomb_count = 3
    index_agent_blast = 4
    index_agent_can_kick = 5
#    index_agent_has_teammate = 6
#    index_agent_mate_or_enemy1_loc = 7
    index_agent_enemy2_loc = 6
    index_agent_enemy3_loc = 7

    index_passage_loc = 8
    index_wall_loc = 9       # binary
    index_wood_loc = 10       # binary
    index_flame_loc = 11      # binary

    index_powerup_extrabomb = 12
    index_powerup_increaseblast = 13
    index_powerup_cankick = 14
    index_game_timestep = 15 # float % of game episode

    index2_agent_loc = 16
    index2_agent_bomb_count = 17
    index2_agent_blast = 18
    index2_agent_can_kick = 19

    #print("current board is \n")
    #print(current_board, "\n")

    alives = np.array(observation['alive'])

    our_agent_pos = np.array(observation['position'])
    #print("our position is ", our_agent_pos)
    our_agentid = my_game_id
    #print("our agent id is ", our_agentid)

    agent_positions = _agents_positions(current_board) # returns all the agent locations

   # print("agent positions are", agent_positions, " \n")

    enemies = np.array(observation['enemies'])

   # print("enemies are ", enemies)


   # print("alive agents are ", alives)

    ret[index_bomb_strength, :, :] = np.array(observation['bomb_blast_strength'])  # done

    #print( " bomb blast strength is ", ret[index_bomb_strength, :, :] )

    ret[index_bomb_t_explode, :, :] = np.array(observation['bomb_life'])  # done

    #print (" bomb time to explode is ", ret[index_bomb_t_explode,:,:])

    # print(f"agent locations {agent_positions} and our aget id {our_agentid}")

    if our_agentid in alives: # otherwise return our agent location empty or noT? TODO
        ret[index_agent_loc,agent_positions[our_agentid-10][0], agent_positions[our_agentid-10][1]] = 1 # binary map for our agent

    #print(" channel for our agent location ",ret[index_agent_loc,:,:])

    ret[index_agent_bomb_count,:,:] = np.ones((board_size,board_size)) * np.array(observation['ammo']) # integer full map

    #print(" channel for our agent ammo size ",ret[index_agent_bomb_count,:,:])

    ret[index_agent_blast,:,:] = np.ones((board_size, board_size)) * np.array(observation['blast_strength']) # integer map full

    #print(" channel for agent blast strength ",ret[index_agent_blast,:,:])

    ret[index_agent_can_kick,:,:] = np.ones((board_size, board_size)) * np.array(observation['can_kick']) # binary map full

    #print(" channel for agent kick ",ret[index_agent_can_kick,:,:])

    diagonalPlayer = diagonalAgentId(our_agentid) # TODO remove lines setting to zeros after testing

    #if diagonalPlayer in alives and agent_positions.get(diagonalPlayer - 10,False):  #Fixed for partially pbservable version
   #     ret[index_agent_mate_or_enemy1_loc, agent_positions[diagonalPlayer - 10][0], agent_positions[diagonalPlayer - 10][1]] = 1  # set location to binary
   #     if np.array(observation['teammate']) != constants.Item.AgentDummy: # team game
   #         ret[index_agent_has_teammate,:,:] = np.ones((board_size, board_size)) # full binary map

    #print("team mate channel is ", ret[index_agent_has_teammate,:,:])
    #print("team mate location channel is ", ret[index_agent_mate_or_enemy1_loc, :, :])

    otherTwoEnemyIds = nonDiagonalAgents(our_agentid) # returns an array with two elements

  #  print("other agents are ", otherTwoEnemyIds)


    # first layer is the smallest id among two
    # second layer is the higher one

    if otherTwoEnemyIds[0] in alives and agent_positions.get(otherTwoEnemyIds[0]-10,False): #Fixed for partially pbservable version
        ret[index_agent_enemy2_loc,agent_positions[otherTwoEnemyIds[0]-10][0], agent_positions[otherTwoEnemyIds[0]-10][1]] = 1 # set location to binary

    if otherTwoEnemyIds[1] in alives and agent_positions.get(otherTwoEnemyIds[1]-10,False): #Fixed for partially pbservable version
        ret[index_agent_enemy3_loc, agent_positions[otherTwoEnemyIds[1]-10][0], agent_positions[otherTwoEnemyIds[1]-10][1]] = 1  # set location to binary

  #  print("enemy2 location channel is ", ret[index_agent_enemy2_loc, :, :])
  #  print("enemy3 location channel is ", ret[index_agent_enemy3_loc, :, :])


    ret[index_passage_loc,:,:] = _passage(current_board) # done
    ret[index_wall_loc,:,:] = _walls(current_board) # done
    ret[index_wood_loc,:,:] = _wood(current_board)  # done
    ret[index_flame_loc,:,:] = _flame(current_board) # done


    #print("passage is ", ret[index_passage_loc,:,:])
    #print("wall is ", ret[index_wall_loc,:,:])
    #print("wood is ", ret[index_wood_loc,:,:])
    #print("flame is ", ret[index_flame_loc,:,:])

    ret[index_powerup_extrabomb,:,:] = _powerup_ExtraBomb(current_board) # done
    ret[index_powerup_increaseblast, :, :] = _powerup_IncRange(current_board) # done
    ret[index_powerup_cankick, :, :] = _powerup_Kick(current_board) # done

    ignoreTimeIndex=False                    #This will partition the timesteps only in two phases
    if ignoreTimeIndex:
        if index_game_timestep<=40:
            ret[index_game_timestep, :, :] = np.zeros((board_size, board_size))
        else:
            if index_game_timestep < 750:
                ret[index_game_timestep, :, :] = np.ones((board_size, board_size))
            else:
                ret[index_game_timestep, :, :] = np.ones((board_size, board_size)) * (
                            time_step / 800.0)  # done TODO 800 game length here
    else:
        ret[index_game_timestep,:,:] = np.ones((board_size,board_size))*(time_step/800.0) # done TODO 800 game length here

    #print("time is ", ret[index_game_timestep,:,:])

    #print("generated channels")
    our_agentid2 = my_game_id2

    if our_agentid2 in alives: # otherwise return our agent location empty or noT? TODO
        ret[index2_agent_loc,agent_positions[our_agentid2-10][0], agent_positions[our_agentid2-10][1]] = 1 # binary map for our agent
        #print(" channel for teammate agent location ",ret[index2_agent_loc,:,:])

        ret[index2_agent_bomb_count,:,:] = np.ones((board_size,board_size)) * np.array(observation2['ammo']) # integer full map

        #print(" channel for teammate agent ammo size ",ret[index2_agent_bomb_count,:,:])

        ret[index2_agent_blast,:,:] = np.ones((board_size, board_size)) * np.array(observation2['blast_strength']) # integer map full

        #print(" channel for teammate agent blast strength ",ret[index2_agent_blast,:,:])

        ret[index2_agent_can_kick,:,:] = np.ones((board_size, board_size)) * np.array(observation2['can_kick']) # binary map full
        #print(" channel for teammate agent kick ",ret[index2_agent_can_kick,:,:])

    return ret



