import math
from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.move import SPECIAL_MOVES
from poke_env.data import POKEDEX
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.double_battle import DoubleBattle
from typing import Dict, List, Any, Tuple

# https://github.com/hsahovic/poke-env/blob/1a35c10648fd99797c0e4fe1eb595c295b4ea8ba/src/poke_env/environment/double_battle.py#L215
def get_possible_showdown_targets(
    battle: DoubleBattle,
    pokemon: Pokemon,
    move: Move,
    self_position: int,
    dynamax: bool = False,
) -> Tuple[List[int], List[int]]:
    r"""
    Custom implementation of the get_possible_showdown_targets method provided
    by poke_env. The advantage is that the current version can also provide
    targets for non-allies.
    """
    # Struggle or Recharge
    if move.id in SPECIAL_MOVES and move.id == "recharge":
        return ([battle.EMPTY_TARGET_POSITION], [battle.EMPTY_TARGET_POSITION])

    map_ally = {-2: -1, -1: -2, 1: 2, 2: 1}
    # identify the ally position
    ally_position = map_ally[self_position]
    # identify the opponent positions
    opponent_positions = [1, 2] if self_position < 0 else [-1, -2]

    if dynamax or pokemon.is_dynamaxed:
        if move.category == MoveCategory.STATUS:
            pareto_targets = [battle.EMPTY_TARGET_POSITION]
            original_targets = [battle.EMPTY_TARGET_POSITION]
        else:
            pareto_targets = opponent_positions
            original_targets = opponent_positions
    elif move.non_ghost_target and (
        PokemonType.GHOST not in pokemon.types
    ):  # fixing target for Curse
        return ([battle.EMPTY_TARGET_POSITION], [battle.EMPTY_TARGET_POSITION])
    else:
        pareto_targets = {
            "adjacentAlly": [ally_position],  # helping hand
            "adjacentAllyOrSelf": [ally_position, self_position],
            "adjacentFoe": opponent_positions,
            "all": [3],  # hail
            "allAdjacent": [3],  # earthquake
            "allAdjacentFoes": [4],  # muddy water
            "allies": [5],  # all but only allies - e.g. life dew
            "allySide": [5],  # all allies, but even when switching - e.g. lightscreen
            "allyTeam": [5],  # all teams (generally all status moves)
            "any": [ally_position, *opponent_positions],
            "foeSide": [4],
            "normal": [ally_position, *opponent_positions],
            "randomNormal": opponent_positions,  # convert back to 0
            "scripted": [6],  # depends on the last pokemon that hit
            "self": [self_position],  # convert back to 0
            battle.EMPTY_TARGET_POSITION: [battle.EMPTY_TARGET_POSITION],
            None: opponent_positions,
        }[move.deduced_target]

        original_targets = {
            "adjacentAlly": [ally_position],
            "adjacentAllyOrSelf": [ally_position, self_position],
            "adjacentFoe": opponent_positions,
            "all": [battle.EMPTY_TARGET_POSITION],
            "allAdjacent": [battle.EMPTY_TARGET_POSITION],
            "allAdjacentFoes": [battle.EMPTY_TARGET_POSITION],
            "allies": [battle.EMPTY_TARGET_POSITION],
            "allySide": [battle.EMPTY_TARGET_POSITION],
            "allyTeam": [battle.EMPTY_TARGET_POSITION],
            "any": [ally_position, *opponent_positions],
            "foeSide": [battle.EMPTY_TARGET_POSITION],
            "normal": [ally_position, *opponent_positions],
            "randomNormal": [battle.EMPTY_TARGET_POSITION],
            "scripted": [battle.EMPTY_TARGET_POSITION],
            "self": [battle.EMPTY_TARGET_POSITION],
            battle.EMPTY_TARGET_POSITION: [battle.EMPTY_TARGET_POSITION],
            None: opponent_positions,
        }[move.deduced_target]

    # use this in the pareto

    # pokemon_ids = set(battle._opponent_active_pokemon.keys())
    # pokemon_ids.update(battle._active_pokemon.keys())
    # player_role, opponent_role = (
    #    (battle.player_role, battle.opponent_role)
    #    if self_position < 0
    #    else (battle.opponent_role, battle.player_role)
    # )
    #
    # targets_to_keep = {
    #    {
    #        f"{player_role}a": -1,
    #        f"{player_role}b": -2,
    #        f"{opponent_role}a": 1,
    #        f"{opponent_role}b": 2,
    #    }[pokemon_identifier]
    #    for pokemon_identifier in pokemon_ids
    # }
    # targets_to_keep.add(battle.EMPTY_TARGET_POSITION)
    # targets = [target for target in targets if target in targets_to_keep]

    return original_targets, pareto_targets


def prepare_pokemon_request(mon: Pokemon, pos: int) -> Dict[str, Any]:
    request: Dict = {}
    # insert name
    request["name"] = get_pokemon_showdown_name(mon)
    # insert species
    request["species"] = mon.species
    # pokemon types
    request["types"] = []
    if mon.types:
        request["types"] = [t.name for t in mon.types if t is not None]
    # pokemon weight
    request["weightkg"] = mon.weight
    # pokemon level
    request["level"] = mon.level
    # pokemon gender
    request["gender"] = None
    if mon.gender:
        request["gender"] = mon.gender.name
    # ability
    request["ability"] = None
    if mon.ability:
        request["ability"] = mon.ability
    # dynamax
    request["isDynamaxed"] = mon.is_dynamaxed
    # item
    request["item"] = None
    if mon.item and mon.item != "unknown_item":
        request["item"] = mon.item
    # boots
    # Target format 'hp', 'at', 'df', 'sa', 'sd', 'sp'
    # Current format "atk": 0, "def": 0, "spa": 0, "spd": 0,"spe": 0,
    request["boosts"] = {}
    request["boosts"]["at"] = mon.boosts["atk"]
    request["boosts"]["df"] = mon.boosts["def"]
    request["boosts"]["sa"] = mon.boosts["spa"]
    request["boosts"]["sd"] = mon.boosts["spd"]
    request["boosts"]["sp"] = mon.boosts["spe"]
    # stats
    request["stats"] = {}
    # TODO current hp value is missing
    request["stats"]["at"] = mon.stats["atk"]
    request["stats"]["df"] = mon.stats["def"]
    request["stats"]["sa"] = mon.stats["spa"]
    request["stats"]["sd"] = mon.stats["spd"]
    request["stats"]["sp"] = mon.stats["spe"]
    # status
    request["status"] = None
    if mon.status:
        request["status"] = mon.status.name
    # toxicCounter
    request["toxicCounter"] = mon.status_counter
    request["curHP"] = mon.current_hp
    if pos > 0:
        request["curHP"] *= compute_opponent_stats("hp", mon)
        request["curHP"] = request["curHP"] // 100
    return request


def get_pokemon_showdown_name(mon: Pokemon):
    return POKEDEX[mon.species]["name"]


def compute_opponent_stats(stat: str, mon: Pokemon):
    opp_stat = 0
    # these computations do not assume anything about EV and natures
    if stat == "hp":
        opp_stat = (
            math.floor(((2 * mon.base_stats[stat] + 31) * mon.level) / 100)
            + mon.level
            + 10
        )
    else:
        # for this moment *1 is useless could be used for the nature
        opp_stat = math.floor(
            (math.floor(((2 * mon.base_stats[stat] + 31) * mon.level) / 100) + 5) * 1
        )
    return opp_stat


def compute_initial_stats(mon: Pokemon) -> Dict[str, int]:
    stats_name = ["hp", "atk", "def", "spa", "spd", "spe"]
    stats: Dict[str, int] = {}
    for stat in stats_name:
        stats[stat] = compute_opponent_stats(stat, mon)
    return stats


def does_anybody_have_tabu_moves(battle: DoubleBattle, tabus: List[str]):
    for mon in battle.team.values():
        if mon:
            for move in mon.moves.values():
                if move._id in tabus:
                    return True
    return False


def is_anyone_someone(battle: DoubleBattle, monsters: List[str]):
    for mon in battle.team.values():
        if mon:
            if mon.species in monsters:
                return True
    return False
