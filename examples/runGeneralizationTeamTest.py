"""Run a battle among agents.
"""
import atexit
import random
import time

import argparse
import numpy as np

import sys

sys.path.insert(1, '/home/bilal/anaconda3/envs/playground')

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pommerman import helpers
from pommerman import make
from pommerman import board_generator
import json
from random import randrange
from pommerman import agents as agents
import copy

training_agent_filename="TEAM_ALT_AGENT" ##File name

training_agent = agents.bilal_ccritic_bignnAgent()       #Teammate1
training_agent_2 = agents.bilal_ccritic_bignnAgent()      #Teammate2

num_games_per_opponents = 100    #How many games per set of opponents

shuffle = True
opponents_filename = "Sta_RndNB_RulRnd_RulRndNB_Simple_SimpleNB"
opponent_lists=6
agentsOpp=[[0] for i in range(opponent_lists)]
agentsOpp[0] = [ ## Lists of agents to test, for now, assuming our agents are positions 0 and 2
    training_agent,
    agents.StaticAgent(),
    training_agent_2,
    agents.StaticAgent()
]
agentsOpp[1] = [
    training_agent,
    agents.RandomAgentNoBombs(), training_agent_2, agents.RandomAgentNoBombs()
]
agentsOpp[2] = [
    training_agent,
    agents.RulesRandomAgent(), training_agent_2, agents.RulesRandomAgent()
]
agentsOpp[3] = [
    training_agent,
    agents.RulesRandomAgentNoBomb(), training_agent_2, agents.RulesRandomAgentNoBomb()
]
agentsOpp[4] = [
    training_agent,
    agents.SimpleAgent(), training_agent_2, agents.SimpleAgent()
]
agentsOpp[5] = [
    training_agent,
    agents.SimpleAgentNoBombs(), training_agent_2, agents.SimpleAgentNoBombs()
]


def run(args, seed=None):
    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file
    render_mode = args.render_mode

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.

    typeGame = 'constants.GameType.Team'

    agentsPlay=copy.deepcopy(agentsOpp[0])
    generatedBoard=True

   # NUM_RIGID = 36
   # NUM_WOOD = 36
   # NUM_ITEMS = 20
    if generatedBoard:
        size = 11
        step_count = 0
        rigid = 36
        wood = 36
        items = 20
#        rigid = 0
 #       wood = 0
 #       items = 0
        n_opponents=3
        kick = False
        ammo = 1
        blast = 2
        env = board_generator.randomizeWithTeamAgents(config,agentsPlay, size, rigid, wood, items, step_count, n_opponents,kick,ammo,blast)
        with open("_board_a3ct.txt", 'w') as file:
            info = env.get_json_info()
            file.write(json.dumps(info, sort_keys=True, indent=4))
    else:
        env = make(config, agents, game_state_file)


    if record_pngs_dir and not os.path.isdir(record_pngs_dir):
        os.makedirs(record_pngs_dir)
    if record_json_dir and not os.path.isdir(record_json_dir):
        os.makedirs(record_json_dir)


    def _run(seed, record_pngs_dir=None, record_json_dir=None):
#        env.seed(seed)

        #      print("Starting the Game.")
        obs = env.reset()
        steps = 0
        done = False
        while not done:
            steps += 1
            if args.render:
                env.render(mode=args.render_mode,record_pngs_dir=record_pngs_dir,record_json_dir=record_json_dir)
            actions = env.act(obs)
            obs, reward, done, info = env.step(actions)
           # time.sleep(1)

        for agent in agentsPlay:
     #       print('episode end',agent.agent_id)
            agent.episode_end(reward[agent.agent_id])

    #    print('N steps ', steps, ' Final Result: ', info, str(reward[0]))
        if args.render:
            env.render(mode=args.render_mode)
            env.render(close=True)

        return steps,info, reward[0]

    N = 10
    countWins=0
    countDraws=0
    countLoss=0
    infos = []
    rewAll=[]
    counterAll=0
    timesteps_all=[]
    times = [ []for i in range(opponent_lists)]
    timesteps = [[] for i in range(opponent_lists)]
    rew = [ [] for i in range(opponent_lists)]
    winrate = [ i for i in range(opponent_lists)]
    drawsTotal = [i for i in range(opponent_lists)]

    # print(rew)
    for j in range (len(agentsOpp)):
        agent_list = copy.deepcopy(agentsOpp[j])
        print("Testing agents:"+str(j))
        for id, agent in enumerate(agent_list):
            agent.init_agent(id, typeGame)
        countWins=0
        countDraws=0
        env.set_agents(agent_list)
        for i in range(num_games_per_opponents):
          #  env = board_generator.shuffle(env,config,size,rigid,wood,items,step_count,n_opponents)
          #  env._step_count = step_count

    #        x = randrange(2, rigid, 2)
    #        y = randrange(4, wood, 2)
    #        z = randrange(2, y, 2)
            if shuffle and generatedBoard:
                      env = board_generator.shuffleTeam(env, config, size, rigid, wood, items,
                                                    step_count, n_opponents)

          #  print(i,generatedBoard,shuffle,randomize_tile_size,rigid, wood, items)#x,y,z)
            start = time.time()
            record_pngs_dir_ = record_pngs_dir + '/%d' % (i + 1) \
                if record_pngs_dir else None
            record_json_dir_ = record_json_dir + '/%d' % (i + 1) \
                if record_json_dir else None
            # infos.append(_run(seed, record_pngs_dir_, record_json_dir_))
            steps,inf, r = _run(seed, record_pngs_dir_, record_json_dir_)
            infos.append(inf)
            rew[j].append(r)
            rewAll.append(r)
            if r==1:
                countWins+=1
            elif r==-1 and steps==801:
                countDraws+=1
            else:
                countLoss+=1
         #   print(i,r, steps)

            timesteps[j].append(steps)
            timesteps_all.append(steps)
            times[j].append(time.time() - start)

            winrate[j] = countWins
            drawsTotal[j]=countDraws
            counterAll+=1

    #    moving_rew = np.convolve(rew[j], np.ones((N,)) / N, mode='valid')
    #    moving_timesteps = np.convolve(timesteps[j], np.ones((N,)) / N, mode='valid')
    #    np.savetxt("_r_" + str(j) + "_" + filename + ".txt", rew[j])
    #    np.savetxt("_t_" + str(j) + "_" + filename + ".txt", timesteps[j])

   # moving_rew_all = np.convolve(rewAll, np.ones((N,)) / N, mode='valid')
   # moving_timesteps_all = np.convolve(timesteps_all, np.ones((N,)) / N, mode='valid')
   # np.savetxt("_r_all" + filename + ".txt", rewAll)
  #  np.savetxt("_t_all" + filename + ".txt", timesteps_all)
    atexit.register(env.close)
    np.savetxt("_winrate_" +training_agent_filename+"_"+opponents_filename + ".txt", winrate)
    np.savetxt("_draws_" + training_agent_filename + "_" + opponents_filename + ".txt", drawsTotal)

    return infos

def main():


    parser = argparse.ArgumentParser(description='Playground Flags.')
    parser.add_argument('--config',
                        default='PommeTeamCompetition-v0',
                        help='Configuration to execute. See env_ids in '
                             'configs.py for options.')
    parser.add_argument('--agents',
                    #    default='' + a3c_om_agent + ',' + rules_random + ',' + rules_random + ',' + rules_random,
                        help='Comma delineated list of agent types and docker '
                             'locations to run the agents.')
    parser.add_argument('--agent_env_vars',
                        help='Comma delineated list of agent environment vars '
                             'to pass to Docker. This is only for the Docker Agent.'
                             " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
                             'would send two arguments to Docker Agent 0 and one '
                             'to Docker Agent 3.',
                        default="")
    parser.add_argument('--record_pngs_dir',
                        default=None,
                        help='Directory to record the PNGs of the game. '
                             "Doesn't record if None.")
    parser.add_argument('--record_json_dir',
                        default=None,
                        help='Directory to record the JSON representations of '
                             "the game. Doesn't record if None.")
    parser.add_argument('--render',
                        default=False,
                        action='store_true',
                        help="Whether to render or not. Defaults to False.")
    parser.add_argument('--render_mode',
                        default='human',
                        help="What mode to render. Options are human, rgb_pixel, and rgb_array")
    parser.add_argument('--game_state_file',
                        default=None,
                        help="File from which to load game state.")
    args = parser.parse_args()
    run(args, 1)

if __name__ == "__main__":
    main()
