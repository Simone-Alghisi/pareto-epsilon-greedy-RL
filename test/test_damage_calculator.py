import unittest
import json
import test.helper as helper
from poke_env.environment.pokemon import Pokemon
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon
from poke_env.utils import to_id_str, compute_raw_stats
from poke_env.environment.pokemon_gender import PokemonGender


class TestDamageCalculator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Initiate the two pokemons:

        Attacker:
        Bulbasaur @ Eviolite
        Level: 5
        Serious Nature
        Ability: Mummy
        - Giga Drain
        - Leech Seed
        - Sludge Bomb
        - Earthquake

        Defender:
        Magikarp @ Life Orb
        Level: 5
        Serious Nature
        Ability: Rattled
        - Splash
        - Flail
        - Tackle
        - Bounce
        """
        super(TestDamageCalculator, self).__init__(*args, **kwargs)

        # initiate the attacker
        self.attacker = Pokemon(species="bulbasaur")

        self.attacker._item = "eviolite"
        self.attacker._ability = "Mummy"
        self.attacker._moves = ["giga drain", "leech seed", "sludge bomb", "earthquake"]
        self.attacker._gender = PokemonGender.from_request_details("M")
        self.attacker._shiny = False
        self.attacker._level = 5

        # initiate the defender
        self.defender = Pokemon(species="magikarp")

        self.defender._item = "life orb"
        self.defender._ability = "Rattled"
        self.defender._moves = ["splash", "flail", "tackle", "bounce"]
        self.defender._gender = PokemonGender.from_request_details("M")
        self.defender._shiny = False
        self.defender._level = 5

    def setUp(self):
        """
        Cleans the boosts and the items of the Pokemon.
        The method is the classic beforeEach of every test suite
        """
        # ========= before each test ====================
        # clean the boosts
        self.attacker = helper.clear_boosts(self.attacker)
        self.defender = helper.clear_boosts(self.defender)
        # clean items
        self.attacker._item = "eviolite"
        self.defender._item = "life orb"

    # ============== CLEAN DAMAGE ========================

    def test_clean_damage_single(self):
        """
        Test wether the damage without boosts or item is correct [single battle]
        """
        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Singles"}
            )
        )
        self.assertEqual(
            sium[0]["0"]["damage"], [7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9]
        )

    def test_clean_damage_doubles(self):
        """
        Test wether the damage without boosts or item is correct [double battle]
        """
        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Doubles"}
            )
        )

        self.assertEqual(
            sium[0]["0"]["damage"], [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7]
        )

    # ============== BOST DAMAGE ========================

    def test_boost_damage_single(self):
        """
        Test wether the damage with boosts only is correct [single battle]
        """
        self.attacker = helper.add_boosts(
            self.attacker, helper.create_stats(0, 2, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender, helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Singles"}
            )
        )

        self.assertEqual(
            sium[0]["0"]["damage"],
            [9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11],
        )

    def test_boost_damage_doubles(self):
        """
        Test wether the damage with boosts only is correct [double battle]
        """
        self.attacker = helper.add_boosts(
            self.attacker, helper.create_stats(0, 2, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender, helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Doubles"}
            )
        )

        self.assertEqual(
            sium[0]["0"]["damage"], [6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8]
        )

    # ============== ITEM DAMAGE ========================

    def test_item_damage_single(self):
        """
        Test wether the damage with items only is correct [single battle]
        """
        self.defender._item = "eviolite"

        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Singles"}
            )
        )

        self.assertEqual(
            sium[0]["0"]["damage"], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6]
        )

    def test_item_damage_doubles(self):
        """
        Test wether the damage with items only is correct [double battle]
        """
        self.defender._item = "eviolite"

        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Doubles"}
            )
        )
        self.assertEqual(
            sium[0]["0"]["damage"], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]
        )

    # ============== BOST + ITEM DAMAGE ========================

    def test_boost_item_damage_single(self):
        """
        Test wether the damage with items and boost is correct [single battle]
        """
        self.defender._item = "eviolite"

        self.attacker = helper.add_boosts(
            self.attacker, helper.create_stats(0, 3, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender, helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Singles"}
            )
        )

        self.assertEqual(
            sium[0]["0"]["damage"], [8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10]
        )

    def test_boost_item_damage_doubles(self):
        """
        Test wether the damage with items and boost is correct [double battle]
        """
        self.defender._item = "eviolite"

        self.attacker = helper.add_boosts(
            self.attacker, helper.create_stats(0, 3, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender, helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, "earthquake", self.defender, {"gameType": "Doubles"}
            )
        )

        self.assertEqual(
            sium[0]["0"]["damage"], [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7]
        ) 


if __name__ == "__main__":
    unittest.main()
