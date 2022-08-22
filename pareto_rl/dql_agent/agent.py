import os
from time import sleep, time_ns
from argparse import ArgumentParser, Namespace
import torch
import wandb
import csv
from math import inf
from tqdm import tqdm
from itertools import count
from pareto_rl.dql_agent.classes.darkr_ai import ReplayMemory
from pareto_rl.dql_agent.classes.pareto_player import StaticTeambuilder
from pareto_rl.dql_agent.classes.player import (
    BaseRLPlayer,
    CombineActionRLPlayer,
    DoubleActionRLPlayer,
)
from poke_env.player_configuration import PlayerConfiguration
from pareto_rl.dql_agent.classes.max_damage_player import DoubleMaxDamagePlayer
from pareto_rl.dql_agent.classes.random_player import DoubleRandomPlayer
from pareto_rl.dql_agent.classes.wrapper_player import WrapperPlayer
from pareto_rl.dql_agent.utils.teams import VGC_1_2VS2 as OPP_TEAM
from pareto_rl.dql_agent.utils.teams import VGC_2_2VS2 as TEAM
# from pareto_rl.dql_agent.utils.teams import VGC_1 as OPP_TEAM
# from pareto_rl.dql_agent.utils.teams import VGC_3_2VS2 as TEAM
from pareto_rl.dql_agent.utils.utils import (
    is_anyone_someone,
    does_anybody_have_tabu_moves,
    get_run_number,
    get_pokemon_list,
    sample_team,
)


def configure_subparsers(subparsers):
    r"""Configure a new subparser for DQL agent.

    Args:
        subparsers: subparser
    """

    """
    Subparser parameters:
    Args:

    """
    parser: ArgumentParser = subparsers.add_parser("rlagent", help="Train/test reinforcement learning")
    parser.add_argument('--test', metavar='RUN_NUMBER', nargs='+', type=int,
                                         help='One or two run numbers. If one, test against DoubleMaxDamagePlayer, if two test against eachother.')
    parser.add_argument('-fc','--fitness_metric', type=str, choices={'winrate','reward'},
                                         default='reward', help='Fitness metric to use in test.')
    parser.set_defaults(func=main)


def fill_memory(player: BaseRLPlayer, memory: ReplayMemory, args):
    player.policy_net.eval()

    if player.current_battle is not None:
        player.reset_env(restart=False)
        player.reset_battles()
        player.start_challenging()

    with tqdm(
        total=args["memory"], desc="Filling memory", unit="transitions"
    ) as prog_bar:
        while len(memory) < args["memory"]:
            observation = torch.tensor(
                player.reset(), dtype=torch.double, device=args["device"]
            )
            state = observation

            if does_anybody_have_tabu_moves(
                player.current_battle, ["transform", "allyswitch"]
            ):
                print(
                    "Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!"
                )
                continue

            if is_anyone_someone(player.current_battle, ["ditto", "zoroark"]):
                print("Damn you three, \nDitto and Zoroark!")
                continue

            for _ in count():
                player.update_pm()
                # Select and perform an action
                action = player.policy(state, i_episode=0)

                if isinstance(player, DoubleActionRLPlayer):
                    observation, reward, done, _ = player.step(
                        player._encode_actions(action.tolist())
                    )
                else:
                    observation, reward, done, _ = player.step(action)
                observation = torch.tensor(
                    observation, dtype=torch.double, device=args["device"]
                )
                reward = torch.tensor([reward], device=args["device"])

                # Observe new state
                if not done:
                    next_state = observation
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)
                prog_bar.update()

                # Move to the next state
                state = next_state
                if done:
                    player.agent._team = StaticTeambuilder(
                        sample_team(args["pokemon_list"])
                    )
                    player.opponent._team = StaticTeambuilder(
                        sample_team(args["opponent_list"])
                    )
                    player.set_opponent(player.opponent)
                    break
            player.step_reset()
            player.episode_reset()


def train(player: BaseRLPlayer, num_episodes: int, args):
    memory = ReplayMemory(args["memory"])
    run_number = get_run_number()
    player.start_challenging()

    if args["fill_memory"]:
        fill_memory(player, memory, args)

    best_winrate = 0
    best_reward = -inf

    # train loop
    for i_episode in tqdm(range(num_episodes), desc="Training", unit="episodes"):
        # games
        episode_info = {"episode": i_episode}
        # Intermediate evaluation
        if (
            i_episode % args["eval_interval"] == 0 and i_episode != 0
        ) or i_episode == num_episodes - 1:
            winrate, reward = eval(player, args["eval_interval_episodes"], **args)
            episode_info["winrate"] = winrate
            episode_info["eval_reward"] = reward
            # Save the model weights in case of improvements
            if winrate > best_winrate:
                best_winrate = winrate
                model_path = os.path.abspath(f"./models/winrate_best_{run_number}.pth")
                torch.save(player.policy_net.state_dict(), model_path)
            if reward > best_reward:
                best_reward = reward
                model_path = os.path.abspath(f"./models/reward_best_{run_number}.pth")
                torch.save(player.policy_net.state_dict(), model_path)

        # Initialize the environment and state
        observation = torch.tensor(
            player.reset(), dtype=torch.double, device=args["device"]
        )
        state = observation

        if does_anybody_have_tabu_moves(
            player.current_battle, ["transform", "allyswitch"]
        ):
            print("Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!")
            wandb.log(episode_info)
            continue

        if is_anyone_someone(player.current_battle, ["ditto", "zoroark"]):
            print("Damn you three, \nDitto and Zoroark!")
            wandb.log(episode_info)
            continue

        episode_cumul_loss = 0
        episode_cumul_reward = 0

        t = 0
        for t in count():
            step_info = episode_info
            # turns
            args["step"] += 1

            player.update_pm()
            # Select and perform an action
            action = player.policy(state, i_episode)

            # if not args['combined_actions']:
            if isinstance(player, DoubleActionRLPlayer):
                observation, reward, done, _ = player.step(
                    player._encode_actions(action.tolist())
                )
            else:
                observation, reward, done, _ = player.step(action)
            observation = torch.tensor(
                observation, dtype=torch.double, device=args["device"]
            )
            reward = torch.tensor([reward], device=args["device"])
            episode_cumul_reward += reward.item()

            # Observe new state
            if not done:
                next_state = observation
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = player.optimize_model(memory)
            if loss is not None:
                episode_cumul_loss += loss.item()

            # log current step
            step_info.update(
                {
                    "step": args["step"],
                    "loss": loss,
                    "eps_threshold": player.eps_threshold,
                    "reward": reward,
                }
            )
            wandb.log(step_info)

            if done:
                player.agent._team = StaticTeambuilder(
                    sample_team(args["pokemon_list"])
                )
                player.opponent._team = StaticTeambuilder(
                    sample_team(args["opponent_list"])
                )
                player.set_opponent(player.opponent)
                break
        player.step_reset()
        player.episode_reset()
        episode_info.update(
            {
                "step": args["step"],
                "ep_loss": episode_cumul_loss / (t + 1),
                "ep_reward": episode_cumul_reward / (t + 1),
                "mem_size": len(memory),
            }
        )
        wandb.log(episode_info)
        # Update the target network, copying all weights and biases in DQN
        if i_episode > 0 and i_episode % args["target_update"] == 0:
            player.update_target()

    player.reset_env()
    model_path = os.path.abspath(f"./models/final_{run_number}.pth")
    torch.save(player.policy_net.state_dict(), model_path)

    return best_winrate, best_reward


def eval(player: BaseRLPlayer, num_episodes: int, **args):
    player.policy_net.eval()

    if player.current_battle is not None:
        player.reset_env(restart=False)
        player.reset_battles()
        player.start_challenging()

    episode_reward = 0.0
    for _ in tqdm(range(num_episodes), desc="Evaluating", unit="episodes"):
        observation = torch.tensor(
            player.reset(), dtype=torch.double, device=args["device"]
        )
        state = observation

        if does_anybody_have_tabu_moves(
            player.current_battle, ["transform", "allyswitch"]
        ):
            print("Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!")
            continue

        if is_anyone_someone(player.current_battle, ["ditto", "zoroark"]):
            print("Damn you three, \nDitto and Zoroark!")
            continue

        episode_cumul_reward = 0.0
        t = 0
        for t in count():
            player.update_pm()
            # Follow learned policy (eps_greedy=False -> never choose random move)
            actions = player.policy(state, eps_greedy=False)

            if isinstance(player, DoubleActionRLPlayer):
                observation, reward, done, _ = player.step(
                    player._encode_actions(actions.tolist())
                )
            else:
                observation, reward, done, _ = player.step(actions)
            observation = torch.tensor(
                observation, dtype=torch.double, device=args["device"]
            )

            episode_cumul_reward += reward

            # Observe new state
            if not done:
                next_state = observation
            else:
                player.agent._team = StaticTeambuilder(
                    sample_team(args["pokemon_list"])
                )
                player.opponent._team = StaticTeambuilder(
                    sample_team(args["opponent_list"])
                )
                player.set_opponent(player.opponent)
                break

            # Move to the next state
            state = next_state
        episode_reward += episode_cumul_reward / (t + 1)
        player.step_reset()
        player.episode_reset()

    print(f"DarkrAI has won {player.n_won_battles} out of {num_episodes} games")
    return player.n_won_battles / num_episodes, episode_reward / num_episodes

def test(player: BaseRLPlayer, num_episodes: int, **args):
    player.policy_net.eval()

    if player.current_battle is not None:
        player.reset_env(restart=False)
        player.reset_battles()
        player.start_challenging()

    lines = []
    lines.append(['mean_reward', 'won', 'n_turns', 'wct'])
    for _ in tqdm(range(num_episodes), desc="Evaluating", unit="episodes"):

        mean_reward = 0
        start = time_ns()

        observation = torch.tensor(
            player.reset(), dtype=torch.double, device=args["device"]
        )
        state = observation

        if does_anybody_have_tabu_moves(
            player.current_battle, ["transform", "allyswitch"]
        ):
            print("Damn you, \nMew!\nAnd to all that can AllySwitch\nDamn you, \ntoo!")
            continue

        if is_anyone_someone(player.current_battle, ["ditto", "zoroark"]):
            print("Damn you three, \nDitto and Zoroark!")
            continue

        t = 0
        for t in count():
            player.update_pm()
            # Follow learned policy (eps_greedy=False -> never choose random move)
            actions = player.policy(state, eps_greedy=False)

            if isinstance(player, DoubleActionRLPlayer):
                observation, reward, done, _ = player.step(
                    player._encode_actions(actions.tolist())
                )
            else:
                observation, reward, done, _ = player.step(actions)
            observation = torch.tensor(
                observation, dtype=torch.double, device=args["device"]
            )
            mean_reward += reward

            # Observe new state
            if not done:
                next_state = observation
            else:
                player.agent._team = StaticTeambuilder(
                    sample_team(args["pokemon_list"])
                )
                player.opponent._team = StaticTeambuilder(
                    sample_team(args["opponent_list"])
                )
                player.set_opponent(player.opponent)
                break

            # Move to the next state
            state = next_state
        # CSV
        wct = time_ns() - start
        lines.append([mean_reward/(t+1), int(player.current_battle.won), t+1, wct])

        player.step_reset()
        player.episode_reset()

    with open(f"./analysis/data/{args['fitness_metric']}_{args['test'][0]}.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerows(lines)
    print(f"DarkrAI has won {player.n_won_battles} out of {num_episodes} games")


def train_handler(args, darkrai_player_config, battle_format, opponent):
    if args["combined_actions"]:
        agent = CombineActionRLPlayer(
            args["input_size"],
            args["hidden_layers"],
            args["n_switches"],
            args["n_moves"],
            args["n_targets"],
            args["exp_rate_start"],
            args["exp_rate_end"],
            args["train_episodes"],
            args["batch_size"],
            args["gamma"],
            args["team"],
            args["lr"],
            args["eps"],
            args["pareto_p"],
            args["pareto_thresh"],
            battle_format=battle_format,
            player_configuration=darkrai_player_config,
            opponent=opponent,
            start_timer_on_battle_start=True,
        )
    else:
        agent = DoubleActionRLPlayer(
            args["input_size"],
            args["hidden_layers"],
            args["n_switches"],
            args["n_moves"],
            args["n_targets"],
            args["exp_rate_start"],
            args["exp_rate_end"],
            args["train_episodes"],
            args["batch_size"],
            args["gamma"],
            args["team"],
            args["lr"],
            args["eps"],
            battle_format=battle_format,
            player_configuration=darkrai_player_config,
            opponent=opponent,
            start_timer_on_battle_start=True,
        )

    # parameters of the run
    args["n_actions"] = agent.n_actions
    args["output_size"] = agent.output_size
    wandb.init(project="DarkrAI", entity="darkr-ai", config=args)

    args.update(
        {
            "device": agent.device,
            "step": 0,
        }
    )

    best_winrate, best_reward = train(agent, args["train_episodes"], args)
    wandb.log({"best_winrate": best_winrate, "best_reward": best_reward})

def test_handler(args, darkrai_player_config, battle_format, opponent):
    # fitness_metric = 'reward'
    net1_path = f"./models/{args['fitness_metric']}_best_{args['test'][0]}.pth"
    net2_path = None
    if len(args["test"]) > 1:
        net2_path = f"./models/{args['fitness_metric']}_best_{args['test'][1]}.pth"
        opponent_config = PlayerConfiguration("DarkrAII", None)
        fake_config = PlayerConfiguration("IDoNotExist", None)
        opp_agent = CombineActionRLPlayer(
            args["input_size"],
            args["hidden_layers"],
            args["n_switches"],
            args["n_moves"],
            args["n_targets"],
            args["exp_rate_start"],
            args["exp_rate_end"],
            args["train_episodes"],
            args["batch_size"],
            args["gamma"],
            args["team"],
            args["lr"],
            args["eps"],
            args["pareto_p"],
            args["pareto_thresh"],
            weights_path=net2_path,
            battle_format=battle_format,
            opponent=opponent,
            player_configuration=fake_config,
        )
        opponent = WrapperPlayer(
            opp_agent, battle_format=battle_format, player_configuration=opponent_config
        )

    if args["combined_actions"]:
        agent = CombineActionRLPlayer(
            args["input_size"],
            args["hidden_layers"],
            args["n_switches"],
            args["n_moves"],
            args["n_targets"],
            args["exp_rate_start"],
            args["exp_rate_end"],
            args["train_episodes"],
            args["batch_size"],
            args["gamma"],
            args["team"],
            args["lr"],
            args["eps"],
            args["pareto_p"],
            args["pareto_thresh"],
            weights_path=net1_path,
            battle_format=battle_format,
            player_configuration=darkrai_player_config,
            opponent=opponent,
            start_timer_on_battle_start=True,
        )
    else:
        agent = DoubleActionRLPlayer(
            args["input_size"],
            args["hidden_layers"],
            args["n_switches"],
            args["n_moves"],
            args["n_targets"],
            args["exp_rate_start"],
            args["exp_rate_end"],
            args["train_episodes"],
            args["batch_size"],
            args["gamma"],
            args["team"],
            args["lr"],
            args["eps"],
            weights_path=net1_path,
            battle_format=battle_format,
            player_configuration=darkrai_player_config,
            opponent=opponent,
            start_timer_on_battle_start=True,
        )

    # parameters of the run
    args["n_actions"] = agent.n_actions
    args["output_size"] = agent.output_size

    args.update(
        {
            "device": agent.device,
            "step": 0,
        }
    )
    agent.start_challenging()
    test(agent,1000,**args)



def main(args):
    args = vars(args)
    args.update({
        "exp_rate_start": 1.0,
        "exp_rate_end": 0.10,
        "train_episodes": 3000,
        "batch_size": 32,
        "gamma": 0.999,
        "team": TEAM,
        "lr": 1e-4,
        "eps": 1e-6,
        "n_switches": 0,
        "n_moves": 4,
        "n_targets": 5,
        "input_size": 124,
        "hidden_layers": [256,128],
        "target_update": 1000,
        "eval_interval": 500,
        "eval_interval_episodes": 300,
        "memory": 32 * 40,
        "combined_actions": True,
        "fixed_team": True,
        "fill_memory": True,
        "pareto_p": 0.7,
        "pareto_thresh": 0.2,
        "pokemon_list": get_pokemon_list([TEAM]),
        "opponent_list": get_pokemon_list([OPP_TEAM]),
    })

    darkrai_player_config = PlayerConfiguration("DarkrAI", None)
    if args["fixed_team"]:
        battle_format = "gen8doublesubers"
        opponent_config = PlayerConfiguration("ThatsALottaDamage", None)
        opponent = DoubleMaxDamagePlayer(
            battle_format=battle_format, player_configuration=opponent_config
        )
    else:
        battle_format = "gen8randomdoublesbattle"
        opponent_config = PlayerConfiguration("RandomMeansRandom", None)
        opponent = DoubleRandomPlayer(
            battle_format=battle_format, player_configuration=opponent_config
        )

    if args['test'] is None:
        train_handler(args, darkrai_player_config, battle_format, opponent)
    else:
        test_handler(args, darkrai_player_config, battle_format, opponent)
