import asyncio
from pareto_rl.dql_agent.classes.pareto_player import ParetoPlayer
from poke_env.player_configuration import PlayerConfiguration
import pareto_rl.pareto_front.ga.utils.plot_utils as plot_utils
from pareto_rl.pareto_front.ga.nsga2 import get_evaluations, parse_evaluation, init_nsga2
from pathlib import Path

# https://github.com/hsahovic/poke-env/blob/master/examples/connecting_an_agent_to_showdown.py
def configure_subparsers(subparsers):
    r"""Configure a new subparser for testing the Pareto agent.

    Args:
      subparsers: subparser
    """

    """
    Subparser parameters:
    Args:
      player [str]: which opponent to duel
    """
    parser = subparsers.add_parser("pareto-battle", help="Test the Pareto agent")
    parser.add_argument(
      "--player", type=str, default="ParetePareteParete", help="Player to challenge"
    )
    parser.add_argument(
      "--dry", action="store_true", default=False, help="Run without showing plots"
    )
    parser.set_defaults(func=main)

def main(args):
  r"""Checks the command line arguments and then runs pareto_battle.
  Args:
    args: command line arguments
  """

  print("\n### Testing Pareto battle ###")
  print("> Parameters:")
  for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print("\t{}: {}".format(p, v))
  print("\n")

  # init nsga2
  init_nsga2()

  asyncio.get_event_loop().run_until_complete(pareto_battle(args))

  # plot the diagrams if the argument dry is not specified
  if not args.dry:
    folder = f"{Path(__file__).parent.absolute()}/../../nsga2_runs/"
    files = get_evaluations(folder)
    populations = []
    for i, file in enumerate(files):
        population, args = parse_evaluation(file)
        populations.append(population)
        plot_utils.generate_multilevel_diagram(population, f"NSGA2 run {i} multilevel diagram")
        plot_utils.plot_results_multi_objective_PF(population, f"NSGA2 run {i} objective vs objective", args)
    plot_utils.generate_multilevel_diagram_different_population(populations, "NSGA2 runs comparisons multilevel diagram")

async def pareto_battle(args):
  r"""Asynchronous function which allows the user passed as an argument to duel with
  a Player which chooses Pareto-optimal moves
  Args:
    args: command line arguments
  """
  p_player = ParetoPlayer(
    player_configuration=PlayerConfiguration("ParetoPlayer", None),
  )
  await p_player.send_challenges(opponent=args.player, n_challenges=1)
