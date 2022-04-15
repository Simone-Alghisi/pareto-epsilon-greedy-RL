from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.move import SPECIAL_MOVES
from poke_env.data import POKEDEX
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.double_battle import DoubleBattle

# https://github.com/hsahovic/poke-env/blob/1a35c10648fd99797c0e4fe1eb595c295b4ea8ba/src/poke_env/environment/double_battle.py#L215
def get_possible_showdown_targets(
	battle: DoubleBattle,
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

	return targets


def get_pokemon_showdown_name(pokemon: Pokemon):
	return POKEDEX[pokemon.species]["name"]
