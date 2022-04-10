import asyncio
from pareto_rl.dql_agent.classes.pareto_player import ParetoPlayer
from poke_env.player_configuration import PlayerConfiguration

# https://github.com/hsahovic/poke-env/blob/master/examples/connecting_an_agent_to_showdown.py
def configure_subparsers(subparsers):
    r"""Configure a new subparser for testing the Pareto agent.

    Args:
      subparsers: subparser
    """

    """
    Subparser parameters:
    Args:
    
    """
    parser = subparsers.add_parser("pareto-battle", help="Test the Pareto agent")
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

  asyncio.get_event_loop().run_until_complete(pareto_battle(args))

# send_challenges only works if the function is async
async def pareto_battle(args):
  p_player = ParetoPlayer(
    player_configuration=PlayerConfiguration("ParetoPlayer", None),
  )
  await p_player.send_challenges(opponent='nextSeason', n_challenges=1)