from poke_env.player.player import Player
from poke_env.environment.move import Move, SPECIAL_MOVES
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import (
    BattleOrder,
)
from poke_env.teambuilder.teambuilder import Teambuilder
from pareto_rl.pareto_front.pareto_search import pareto_search
import random
import json

class ParetoPlayer(Player):
    r"""ParetoPlayer, should perform only Pareto optimal moves"""

    def __init__(self, **kwargs):
        super(ParetoPlayer, self).__init__(
            battle_format="gen82v2doubles", team=StaticTeambuilder(), **kwargs
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        pareto_args = {}

        # your mons
        pos = -1
        for mon, moves in zip(battle.active_pokemon, battle.available_moves):
            if mon:
                targets = {
                    move: self.get_possible_showdown_targets(battle, mon, move, pos)
                    for move in moves
                }
                print(targets)
                pareto_args[pos] = targets
            pos -= 1

        # opponent mons
        pos = 1
        for mon in battle.opponent_active_pokemon:
            if mon:
                if mon.species.strip().lower() == 'Zigzagoon-Galar'.strip().lower():
                    moves = {Move("doubleedge"), Move("surf"), Move('knockoff'), Move('thunderbolt')}
                else:
                    moves = {Move("fireblast"), Move("firefang"), Move('fling'), Move('seismictoss')}
                # in the general case, first retrieve known moves and then infer the other probabilistically
                #moves = mon.moves # pokemon used moves
                targets = {
                    move: self.get_possible_showdown_targets(battle, mon, move, pos)
                    for move in moves
                }
                print(targets)
                pareto_args[pos] = targets
            pos += 1
        
        # remove invalid targets for basic
        for pos, moves in pareto_args.items():
            for m, targets in moves.items():
                tmp_targets = targets.copy()
                for t in targets:
                    if (t not in pareto_args) and (t < 3):
                        tmp_targets.remove(t)
                pareto_args[pos][m] = tmp_targets

        print(pareto_args)
        return self.choose_random_doubles_move(battle)
    
    # https://github.com/hsahovic/poke-env/blob/1a35c10648fd99797c0e4fe1eb595c295b4ea8ba/src/poke_env/environment/double_battle.py#L215
    def get_possible_showdown_targets(
        self,
        battle: AbstractBattle,
        pokemon: Pokemon,
        move: Move,
        self_position: int,
        dynamax: bool = False,
    ):
        # Struggle or Recharge
        if move.id in SPECIAL_MOVES and move.id == "recharge":
            return [battle.EMPTY_TARGET_POSITION]

        map_ally = {-2: -1, -1: -2, 1: 2, 2: 1}
        # identify the ally position
        ally_position = map_ally[self_position]
        # identify the opponent positions
        opponent_positions = [1, 2] if self_position < 0 else [-1, -2]

        if dynamax or pokemon.is_dynamaxed:
            if move.category == MoveCategory.STATUS:
                targets = [battle.EMPTY_TARGET_POSITION]
            else:
                targets = opponent_positions
        elif move.non_ghost_target and (
            PokemonType.GHOST not in pokemon.types
        ):  # fixing target for Curse
            return [battle.EMPTY_TARGET_POSITION]
        else:
            targets = {
                "adjacentAlly": [ally_position],  # helping hand
                "adjacentAllyOrSelf": [ally_position, self_position],
                "adjacentFoe": opponent_positions,
                "all": [3],  # hail
                "allAdjacent": [3],  # earthquake
                "allAdjacentFoes": [4],  # muddy water
                "allies": [5],  # all but only allies - e.g. life dew
                "allySide": [
                    5
                ],  # all allies, but even when switching - e.g. lightscreen
                "allyTeam": [5],  # all teams (generally all status moves)
                "any": [ally_position, *opponent_positions],
                "foeSide": [6],
                "normal": [ally_position, *opponent_positions],
                "randomNormal": opponent_positions,
                "scripted": [7],
                "self": [self_position],
                battle.EMPTY_TARGET_POSITION: [battle.EMPTY_TARGET_POSITION],
                None: opponent_positions,
            }[move.deduced_target]

        pokemon_ids = set(battle._opponent_active_pokemon.keys())
        pokemon_ids.update(battle._active_pokemon.keys())
        player_role, opponent_role = (
            (battle.player_role, battle.opponent_role)
            if self_position < 0
            else (battle.opponent_role, battle.player_role)
        )

        # use this in the pareto

        # targets_to_keep = {
        #    {
        #        f"{player_role}a": -1,
        #        f"{player_role}b": -2,
        #        f"{opponent_role}a": 1,
        #        f"{opponent_role}b": 2,
        #    }[pokemon_identifier]
        #    for pokemon_identifier in pokemon_ids
        #}
        #targets_to_keep.add(battle.EMPTY_TARGET_POSITION)
        #targets = [target for target in targets if target in targets_to_keep]

        return targets


# qui

TEAM = """
Zigzagoon-Galar @ Aguav Berry
Ability: Quick Feet
EVs: 252 HP / 136 Atk / 120 SpA
- Double-Edge
- Surf
- Knock Off
- Thunderbolt

Charmander @ Aguav Berry
Ability: Blaze
EVs: 252 HP / 136 Atk / 120 SpA
- Fire Blast
- Fire Fang
- Fling
- Seismic Toss
"""


class StaticTeambuilder(Teambuilder):
    def yield_team(self):
        team = self.join_team(self.parse_showdown_team(TEAM))
        print(team)
        return team


# https://github.com/hsahovic/poke-env/blob/c5464ab43719af4e24af835c446e15b78c022348/src/poke_env/player/player.py
