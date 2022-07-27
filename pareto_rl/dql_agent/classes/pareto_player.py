import orjson
import random
import types
from argparse import Namespace
from poke_env.environment.pokemon import Pokemon
from poke_env.player.player import Player
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.teambuilder.teambuilder import Teambuilder
from pareto_rl.dql_agent.utils.move import Move
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from pareto_rl.dql_agent.utils.teams import TEST_TEAM_1 as TEAM
from pareto_rl.dql_agent.utils.teams import TEST_TEAM_2 as OPP_TEAM
from pareto_rl.dql_agent.utils.utils import compute_initial_stats
from pareto_rl.pareto_front.pareto_search import pareto_search
from typing import List, Tuple, Dict


class ParetoPlayer(Player):
    r"""ParetoPlayer, should perform only Pareto optimal moves"""

    def __init__(self, battle_format="gen8doublesubers", team=None, **kwargs):
        if team is None:
            team = StaticTeambuilder(TEAM)
        super(ParetoPlayer, self).__init__(
            battle_format=battle_format, team=team, **kwargs
        )
        self.last_turn: List[Tuple[str, str]] = []
        self.estimates: Dict[str, Dict[str, Dict[str, int]]] = {
            "mon": {},
            "opp": {},
        }
        bind_pareto(self)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        pm = PokemonMapper(battle, OPP_TEAM)
        # TODO idk if this can or cannot handle switch properly
        self.analyse_previous_turn(pm, battle)
        if sum(battle.force_switch) > 0:
            return self.choose_random_doubles_move(battle)
        args = Namespace(dry=True)
        orders = pareto_search(args, battle, pm, self)
        return orders[int(random.random() * len(orders))]


def bind_pareto(instance):
    instance.last_turn: List[Tuple[str, str]] = []
    instance.estimates: Dict[str, Dict[str, Dict[str, int]]] = {"mon": {}, "opp": {}}
    instance.get_mon_estimates = types.MethodType(get_mon_estimates, instance)
    instance.update_mon_estimates = types.MethodType(update_mon_estimates, instance)
    instance.analyse_previous_turn = types.MethodType(analyse_previous_turn, instance)
    instance._handle_battle_message = types.MethodType(_handle_battle_message, instance)


def get_mon_estimates(self, mon: Pokemon, pos: int) -> Dict[str, int]:
    category = "mon" if pos < 0 else "opp"
    if mon.species not in self.estimates[category]:
        if pos < 0:
            self.estimates[category][mon.species] = mon.stats
        else:
            self.estimates[category][mon.species] = compute_initial_stats(mon)
    return self.estimates[category][mon.species]


def update_mon_estimates(
    self, mon: Pokemon, pos: int, stat: str, new_value: int
) -> None:
    category = "mon" if pos < 0 else "opp"
    self.estimates[category][mon.species][stat] = new_value


def analyse_previous_turn(self, pm: PokemonMapper, battle: DoubleBattle) -> None:
    r"""
    Returns a possible turn order (prediction) based on the current moves
    to be perfomed in the genotype for the current turn and the estimated
    speed.
    Args:
        - c: a candidate (genotype) encoding moves and target for each
        attacker
        - pm [PokemonMapper]
        - last_turn: the last turn which has been performed (TODO... maybe
        move it in a separate function)
    Returns:
        turn_order [List[int]]: a list encoding the possible turn order
        containing the position of the attacker that will act (from first
        to last)
    """
    map_showdown_to_pos = {"p1a": -1, "p1b": -2, "p2a": 1, "p2b": 2}

    actual_turn = []
    predicted_turn = []

    # for mon_str, move_str in self.last_turn:
    for action in self.last_turn:
        mon_str = action
        if isinstance(action, tuple):
            mon_str = action[0]
        # convert from showdown to pokemon instance
        showdown_pos = mon_str.split(":")[0]
        pos = map_showdown_to_pos[showdown_pos]
        # handle if a mon died
        if pos not in pm.pos_to_mon:
            continue
        mon = pm.pos_to_mon[pos]
        if isinstance(action, tuple):
            move_str = action[1]
            move_id = Move.retrieve_id(move_str)
            move = Move(move_id)
            priority = move.priority
        else:
            # https://pokemondb.net/pokebase/119034/how-much-priority-does-switching-have
            priority = 6
        mon_speed = self.get_mon_estimates(mon, pos)["spe"]
        actual_turn.append((pos, mon, priority, mon_speed))
        predicted_turn.append((pos, mon, priority, mon_speed))

    predicted_turn.sort(key=lambda x: (x[2], x[3]), reverse=True)

    already_examined = set()
    i = 0
    j = 0

    # updating the beliefs concerning the opponent pokemon speed/priority
    while i < len(actual_turn) and j < len(predicted_turn):
        pos, mon, priority, mon_speed = actual_turn[i]
        pr_pos, pr_mon, pr_priority, pr_mon_speed = predicted_turn[j]

        if pr_pos in already_examined:
            # we already adjusted the speed of that pokemon
            j += 1
        elif pos != pr_pos:
            if priority == pr_priority:
                # at least one is an opponent, so I do not know everything about him
                if pos > 0:
                    # consider case where mon it's faster
                    self.update_mon_estimates(mon, pos, "spe", pr_mon_speed + 1)
                elif pos < 0 and pr_pos > 0:
                    self.update_mon_estimates(pr_mon, pr_pos, "spe", mon_speed)
            already_examined.add(pos)
            i += 1
        else:
            # we predicted correctly
            i += 1
            j += 1


async def _handle_battle_message(self, split_messages: List[List[str]]) -> None:
    """Handles a battle message.
    :param split_message: The received battle message.
    :type split_message: str
    """

    # Battle messages can be multiline
    if (
        len(split_messages) > 1
        and len(split_messages[1]) > 1
        and split_messages[1][1] == "init"
    ):
        battle_info = split_messages[0][0].split("-")
        battle = await self._create_battle(battle_info)
    else:
        battle = await self._get_battle(split_messages[0][0])

    for split_message in split_messages[1:]:
        if len(split_message) <= 1:
            continue
        elif split_message[1] in self.MESSAGES_TO_IGNORE:
            pass
        elif split_message[1] == "request":
            if split_message[2]:
                request = orjson.loads(split_message[2])
                battle._parse_request(request)
                if battle.move_on_next_request:
                    await self._handle_battle_request(battle)
                    battle.move_on_next_request = False
        elif split_message[1] == "win" or split_message[1] == "tie":
            if split_message[1] == "win":
                battle._won_by(split_message[2])
            else:
                battle._tied()
            await self._battle_count_queue.get()
            self._battle_count_queue.task_done()
            self._battle_finished_callback(battle)
            async with self._battle_end_condition:
                self._battle_end_condition.notify_all()
        elif split_message[1] == "error":
            self.logger.log(25, "Error message received: %s", "|".join(split_message))
            if split_message[2].startswith(
                "[Invalid choice] Sorry, too late to make a different move"
            ):
                if battle.trapped:
                    await self._handle_battle_request(battle)
            elif split_message[2].startswith(
                "[Unavailable choice] Can't switch: The active Pokémon is " "trapped"
            ) or split_message[2].startswith(
                "[Invalid choice] Can't switch: The active Pokémon is trapped"
            ):
                battle.trapped = True
                await self._handle_battle_request(battle)
            elif split_message[2].startswith(
                "[Invalid choice] Can't switch: You can't switch to an active "
                "Pokémon"
            ):
                await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[2].startswith(
                "[Invalid choice] Can't switch: You can't switch to a fainted "
                "Pokémon"
            ):
                await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[2].startswith(
                "[Invalid choice] Can't move: Invalid target for"
            ):
                await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[2].startswith(
                "[Invalid choice] Can't move: You can't choose a target for"
            ):
                await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[2].startswith(
                "[Invalid choice] Can't move: "
            ) and split_message[2].endswith("needs a target"):
                await self._handle_battle_request(battle, maybe_default_order=True)
            elif (
                split_message[2].startswith("[Invalid choice] Can't move: Your")
                and " doesn't have a move matching " in split_message[2]
            ):
                await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[2].startswith("[Invalid choice] Incomplete choice: "):
                await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[2].startswith("[Unavailable choice]") and split_message[
                2
            ].endswith("is disabled"):
                battle.move_on_next_request = True
            elif split_message[2].startswith(
                "[Invalid choice] Can't move: You sent more choices than unfainted"
                " Pokémon."
            ):
                await self._handle_battle_request(battle, maybe_default_order=True)
            else:
                self.logger.critical("Unexpected error message: %s", split_message)
        elif split_message[1] == "turn":
            battle._parse_message(split_message)
            await self._handle_battle_request(battle)
        elif split_message[1] == "teampreview":
            battle._parse_message(split_message)
            await self._handle_battle_request(battle, from_teampreview_request=True)
        elif split_message[1] == "bigerror":
            self.logger.warning("Received 'bigerror' message: %s", split_message)
        else:
            battle._parse_message(split_message)

        # extract all relevant information to exploit the last turn knowledge
        if split_message[1] == "switch":
            # append the actual order of the pokemon to last turn
            switch = split_message[2]
            self.last_turn.append(switch)
        elif split_message[1] == "move":
            # append the actual order of the pokemon to last turn and their move
            mon = split_message[2]
            move = split_message[3]
            self.last_turn.append((mon, move))


class StaticTeambuilder(Teambuilder):
    def __init__(self, team, **kwargs):
        super(StaticTeambuilder, self).__init__(**kwargs)
        self.team = team

    def yield_team(self):
        team = self.join_team(self.parse_showdown_team(self.team))
        return team


# https://github.com/hsahovic/poke-env/blob/c5464ab43719af4e24af835c446e15b78c022348/src/poke_env/player/player.py
