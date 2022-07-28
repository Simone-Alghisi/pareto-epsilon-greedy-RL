import pareto_rl.damage_calculator.requester as rq
from poke_env.environment.pokemon import Pokemon
from pareto_rl.dql_agent.utils.utils import prepare_pokemon_request
from poke_env.utils import to_id_str, compute_raw_stats
from typing import List

def create_stats(
    accuracy: int, atk: int, defence: int, evasion: int, spa: int, spd: int, spe: int
) -> dict:
    """
    Given a series of statistics, returns a dictionary with statistic-type as a key and it's value
    """
    return {
        "accuracy": accuracy,
        "atk": atk,
        "def": defence,
        "evasion": evasion,
        "spa": spa,
        "spd": spd,
        "spe": spe,
    }


def add_boosts(mon: Pokemon, boosts: dict):
    """
    Update Pokemon s boosts with the given dictionary
    """
    mon._boosts = boosts
    return mon


def clear_boosts(mon: Pokemon):
    """
    Nullifies all boosts of a pokemon
    """
    return add_boosts(mon, create_stats(0, 0, 0, 0, 0, 0, 0))


def compute_stats(mon: Pokemon, ivs: List[int], evs: List[int], nature: str):
    """
    Compute raw stats with ivs, evs and nature
    """
    raw = compute_raw_stats(mon.species, ivs, evs, mon._level, nature)
    return {"atk": raw[0], "def": raw[1], "spa": raw[2], "spd": raw[3], "spe": raw[4]}


def make_request(attacker: Pokemon, move: str, defender: Pokemon, field: dict) -> str:
    """
    Simple request to the server

    Args:
      attacker [Pokemon]: pokemon which attacks
      move [str]: name of the move employed
      defender [Pokemon]: pokemon which defend
      field [dict]: SIUM field
    """
    singleton_request = {
        0: {
            "attacker": prepare_pokemon_request(attacker),
            "move": move,
            "target": prepare_pokemon_request(defender),
            "field": field,
        },
    }
    # prepare the request
    request = {"requests": [singleton_request]}
    # return after sending the request and await
    return rq.damage_request_server(request)