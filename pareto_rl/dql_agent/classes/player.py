from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.player.env_player import Gen8EnvSinglePlayer
from gym.spaces import Space
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
)
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from pareto_rl.dql_agent.utils.move import Move
from pareto_rl.dql_agent.utils.pokemon_mapper import PokemonMapper
from typing import Union


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, **kwargs):
        super(SimpleRLPlayer, self).__init__(**kwargs)

    def embed_battle(self, battle: DoubleBattle) -> list:
        obs = []
        active = []
        bench = []

        for mon in battle.team.values():
            # lots of info are available, the problem is time,
            # hughes effect, and also mapping
            mon_data = []

            # types (2)
            types = [t.value if t is not None else -1 for t in mon.types]
            mon_data.extend(types)

            # hp normalised (good idea?)
            mon_data.append(mon.current_hp_fraction)

            # stats (5)
            mon_data.extend(list(mon.stats.values()))

            # boosts and debuffs (7)
            # TODO it may be possible to compute it together with
            # the stats above to reduce the parameters
            mon_data.extend(list(mon.boosts.values()))

            # status
            # TODO one-hot-encoding?
            mon_data.append(mon.status.value if mon.status is not None else -1)

            # moves
            # TODO... is it possible to have less than 4?
            for move in mon.moves.values():
                move_data = []

                # TODO... should we consider insering the move id?
                # while it may be difficult to learn...
                # it may be particularly useful to discriminate
                # the final effect of the move if similar
                # N.B this is a string, need to convert it using
                # the MOVES dictionary from the json
                # move_data.append(move._id)

                # base power
                move_damage = move.base_power
                # consider STAB (same type attack bonus)
                if move.type in mon.types:
                    move_damage *= 1.5
                move_data.append(move_damage)

                # priority
                move_data.append(move.priority)

                # accuracy
                # TODO... should we encode together w/ damage?
                move_data.append(move.accuracy)

                # category (?)
                move_data.append(move.category.value)

                # pp (?)
                # move_data.append(move.current_pp / move.max_pp)

                # recoil (?)
                # move_data.append(move.recoil)

                # damage for each active opponent (2)
                for opp in battle.opponent_active_pokemon:
                    if opp is not None:
                        mlt = move.type.damage_multiplier(opp.type_1, opp.type_2)
                        move_data.append(move_damage * mlt)
                    else:
                        # if one is dead, append -1
                        move_data.append(-1)

                mon_data.extend(move_data)

            mon_data.extend([-1 for _ in range(6)] * (4 - len(mon.moves)))

            if mon.active == True:
                active.extend(mon_data)
            else:
                bench.extend(mon_data)

        obs.extend(active)
        obs.extend(bench)

        # we could also take into account the opponents
        # (which I would say is mandatory)
        # and the field conditions (at least some of them)
        return obs

    def describe_embedding(self) -> Space:
        return super().describe_embedding()

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            last_battle, fainted_value=2, hp_value=1, victory_value=30
        ) - self.reward_computing_helper(
            current_battle, fainted_value=2, hp_value=1, victory_value=30
        )
        # return self.reward_computing_helper(current_battle,fainted_value=2,hp_value=1,victory_value=30)

    def get_pokemon_order(self, action, idx, battle: DoubleBattle) -> BattleOrder:
        poke_mapper = PokemonMapper(battle)
        idx_to_pos = {0: -1, 1: -2}
        pos = idx_to_pos[idx]
        n_targets = 5
        n_switches = 4
        mon_actions = 4 * n_targets + n_switches
        if pos in poke_mapper.moves_targets:
            moves = [move for move in poke_mapper.moves_targets[pos].keys()]
        if action < (mon_actions - n_switches) and not battle.force_switch[idx]:
            # moves
            move = action // n_targets
            # target = DoubleBattle.OPPONENT_1_POSITION if action % 2 == 0 else DoubleBattle.OPPONENT_2_POSITION
            target = (action % n_targets) - 2
            if move >= len(moves):
                import pdb

                pdb.set_trace()
            return self.agent.create_order(moves[move], move_target=target)
        elif action >= (mon_actions - n_switches) and not battle.force_switch[idx]:
            switch = action - (mon_actions - n_switches)
            # print(idx, switch, battle.available_switches)
            return self.agent.create_order(battle.available_switches[idx][switch])
        else:
            return self.agent.choose_random_move(battle)

    def action_to_move(
        self, actions, battle: DoubleBattle
    ) -> BattleOrder:  # pyre-ignore
        """Converts actions to move orders.
        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        actions = decode_actions(actions)
        if actions[0] == -1 or actions[1] == -1:
            return ForfeitBattleOrder()
        battle_order = None
        if battle.force_switch[0] or battle.force_switch[1]:
            battle_order = DoubleBattleOrder(
                self.agent.choose_random_move(battle), None
            )
        else:
            first_order = None
            second_order = None
            for i, mon in enumerate(battle.active_pokemon):
                if mon:
                    if first_order is None:
                        first_order = self.get_pokemon_order(actions[i], i, battle)
                    else:
                        second_order = self.get_pokemon_order(actions[i], i, battle)
            battle_order = DoubleBattleOrder(first_order, second_order)

        # print(battle_order)
        return battle_order


# def decode_actions(coded_action):
#    action1 = coded_action // 100
#    action2 = coded_action % 100
#    return [action1, action2]


def decode_action(
    neuron_pos: int,
    n_actions: int,
    n_switches: int,
    n_moves: int,
    n_targets,
    pm: PokemonMapper,
) -> Union[DoubleBattleOrder, DefaultBattleOrder]:
    if neuron_pos < n_actions * (n_moves * n_targets):
        # the first one wants to make a move
        move_1 = neuron_pos // n_actions  # [0, n_targets*n_moves)
        move_1_idx = move_1 // n_targets  # [0, n_moves)
        move_1_trg = move_1 % n_targets  # [0, n_targets)
        act_2 = neuron_pos % n_actions  # [0, n_actions)
        if act_2 < (n_moves * n_targets):  # [0, n_actions-n_switches), i.e. move
            # the other wants to make a move
            move_2_idx = act_2 // n_targets  # [0, n_moves)
            move_2_trg = act_2 % n_targets  # [0, n_targets)
            # search in available moves
        else:
            # the other wants to switch
            switch_2_trg = act_2 - (n_moves * n_targets)  # [0, n_switches)
            # search in available switches
    elif neuron_pos < n_actions * (n_moves * n_targets) + (
        (n_actions - 1) * n_switches
    ):
        switch = neuron_pos - (n_actions * (n_moves * n_targets))
        switch_1_trg = switch // (n_actions - 1)  # [0, n_switches)
        act_2 = switch % (n_actions - 1)  # [0, n_actions-1)
        if act_2 < (n_moves * n_targets):  # i.e. a move
            move_2_idx = act_2 // n_targets  # [0, n_moves)
            move_2_trg = act_2 % n_targets  # [0, n_targets)
            # search in available moves
        else:
            possible_switches = [
                i for i in range(0, n_switches) if i != switch_1_trg
            ]  # a list with only the possible target switches
            switch_2_trg = possible_switches[
                act_2 - (n_moves * n_targets)
            ]  # [0, n_switches-1), but removing the same
            # search in available switches
    elif (
        neuron_pos
        < n_actions * (n_moves * n_targets)
        + ((n_actions - 1) * n_switches)
        + 2 * n_actions
    ):
        action = (
            neuron_pos
            - n_actions * (n_moves * n_targets)
            + ((n_actions - 1) * n_switches)
        )
        act = action % n_actions
        if action // n_actions == 0:
            # first pokemon
            if act < (n_moves * n_targets):
                # the first pokemon wants to perform a move
                move_1_idx = act // n_targets
                move_1_trg = act % n_targets
                # search in the first pokemon available moves
            else:
                # the first pokemon wants to perform a switch
                switch_1_trg = act - (n_moves * n_targets)
                # search in the first pokemon available switches
        else:
            # second pokemon
            if act < (n_moves * n_targets):
                # the second pokemon wants to perform a move
                move_2_idx = act // n_targets
                move_2_trg = act % n_targets
                # search in the second pokemon available moves
            else:
                # the second pokemon wants to perform a switch
                switch_2_trg = act - (n_moves * n_targets)
                # search in the second pokemon available switches
    else:
        return DefaultBattleOrder()


def encode_action(
    order: Union[BattleOrder, DoubleBattleOrder, DefaultBattleOrder],
    n_actions: int,
    n_switches: int,
    n_moves: int,
    n_targets,
    pm: PokemonMapper,
) -> int:
    if isinstance(order, DefaultBattleOrder):
        return (
            n_actions * (n_moves * n_targets)
            + (n_actions - 1) * n_switches
            + 2 * n_actions
            + 1
        )
    elif isinstance(order, BattleOrder):
        idx = 0
        # TODO we should check if the pokemon is the first or the second
    else:
        first_order = order.first_order
        second_order = order.second_order
        # we need to handle the case in which we only have a single order
        idx = 0
        if isinstance(first_order, Move):
            ...
        elif isinstance(first_order, Pokemon):
            ...
        else:
            print("i don't know what to do")
            exit(1)
