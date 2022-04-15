from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.utils.move import Move
from poke_env.environment.pokemon import Pokemon
from pareto_rl.dql_agent.utils.utils import get_possible_showdown_targets
from typing import Dict, List, Set, Union

class PokemonMapper():
	def __init__(self, battle:DoubleBattle) -> None:
		self.battle = battle
		self.moves_targets: Dict[int, Dict[Move, List[int]]] = {}
		self.mon_to_pos: Dict[Pokemon, int] = {}
		self.pos_to_mon: Dict[int, Pokemon] = {}
		self.mon_indexes: List[int] = []

		# your mons
		pos = -1
		for mon, moves in zip(battle.active_pokemon, battle.available_moves):
			if mon:
				casted_moves: Set[Move] = { Move(move._id) for move in moves }
                # map pokemons with their position
				self.mapper(casted_moves, mon, pos)
			pos -= 1

		# opponent mons
		pos = 1
		for mon in battle.opponent_active_pokemon:
			if mon:
				# hardcoded opponent moves
				moves: Set[Move]
				if mon.species.strip().lower() == "Zigzagoon".strip().lower():
					moves = {
						Move("doubleedge"),
						Move("surf"),
						Move("bodyslam"),
						Move("thunderbolt"),
					}
				else:
					moves = {
						Move("energyball"),
						Move("gigadrain"),
						Move("knockoff"),
						Move("leafstorm"),
					}
				# in the general case, first retrieve known moves and then infer the other probabilistically
				# moves = mon.moves # pokemon used moves
				self.mapper(moves, mon, pos)
			pos += 1

		# remove invalid targets for basic
		for pos, moves in self.moves_targets.items():
			for m, targets in moves.items():
				tmp_targets = targets.copy()
				for t in targets:
					if (t not in self.moves_targets) and (t < 3):
						tmp_targets.remove(t)
				self.moves_targets[pos][m] = tmp_targets

	def mapper(self, moves: Set[Move], mon: Pokemon, pos: int) -> None:
    	# get moves target
		targets: Dict[Move, List[int]] = {
			move: get_possible_showdown_targets(self.battle, mon, move, pos)
			for move in moves
		}
    	# map the target with its position
		self.moves_targets[pos] = targets
		self.mon_to_pos[mon] = pos
		self.pos_to_mon[pos] = mon
		self.mon_indexes.append(pos)

	def alive_pokemon_number(self):
		return len(self.moves_targets)

	def get_field_pos_from_genotype(self, index: int) -> int:
		return self.mon_indexes[index // 2]