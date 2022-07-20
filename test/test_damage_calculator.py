import unittest
import json
import test.helper as helper
from poke_env.environment.pokemon import Pokemon
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon
from poke_env.utils import to_id_str, compute_raw_stats
from poke_env.environment.pokemon_gender import PokemonGender

'''
Bulbasaur @ Eviolite
Level: 5
Calm Nature
Ability: Overgrow
EVs: 252 Atk / 236 SpD / 252 Spe
- Giga Drain
- Leech Seed
- Sludge Bomb
- Toxic

Wingull @ Life Orb
Level: 5
Timid Nature
Ability: Hydration
EVs: 36 HP / 236 SpA / 236 Spe
- Scald
- Hurricane
- U-turn
- Knock Off

'''
class TestDamageCalculator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        '''
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
        Magikarp @ Eviolite
        Level: 5
        Serious Nature
        Ability: Rattled
        - Splash
        - Flail
        - Tackle
        - Bounce
        '''
        super(TestDamageCalculator, self).__init__(*args, **kwargs)

        # initiate the attacker
        self.attacker = Pokemon (
            species = 'bulbasaur'
        )

        self.attacker._item = "eviolite"
        self.attacker._ability = "mummy"
        self.attacker._moves = ["giga drain", "leech seed", "sludge bomb", "earthquake"]
        self.attacker._gender = PokemonGender.from_request_details("M")
        self.attacker._shiny = False
        self.attacker._level = 5

        # initiate the defender
        self.defender = Pokemon(
            species = 'magikarp'
        )

        self.defender._item = "life orb"
        self.defender._ability = "rattled"
        self.defender._moves = ["splash", "flail", "tackle", "bounce"]
        self.defender._gender = PokemonGender.from_request_details("M")
        self.defender._shiny = False
        self.defender._level = 5
    
    def setUp(self):
        '''
        Cleans the boosts and the items of the Pokemon
        '''
        # ========= before each test ====================
        # clean the boosts
        self.attacker = helper.clear_boosts(self.attacker)
        self.defender = helper.clear_boosts(self.defender)
        # clean items
        self.attacker._item = "eviolite"
        self.defender._item = "life orb"

    # ============== CLEAN DAMAGE ========================

    def test_clean_damage_single(self):
        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Singles' }
            )
        )
        self.assertEqual(sium[0]['0']['damage'], [7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9])

    def test_clean_damage_doubles(self):
        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Doubles' }
            )
        )

        self.assertEqual(sium[0]['0']['damage'], [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7])

    # ============== BOST DAMAGE ========================

    def test_boost_damage_single(self):
        self.attacker = helper.add_boosts(
            self.attacker,
            helper.create_stats(0, 2, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender,
            helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Singles' }
            )
        )

        self.assertEqual(sium[0]['0']['damage'], [9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11])

    def test_boost_damage_doubles(self):
        self.attacker = helper.add_boosts(
            self.attacker,
            helper.create_stats(0, 2, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender,
            helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Doubles' }
            )
        )

        self.assertEqual(sium[0]['0']['damage'], [6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8])

    # ============== ITEM DAMAGE ========================

    def test_item_damage_single(self):
        self.defender._item = "eviolite"

        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Singles' }
            )
        )
        
        self.assertEqual(sium[0]['0']['damage'], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6])

    def test_item_damage_doubles(self):
        self.defender._item = "eviolite"

        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Doubles' }
            )
        )
        self.assertEqual(sium[0]['0']['damage'], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4])

    # ============== BOST + ITEM DAMAGE ========================

    def test_boost_item_damage_single(self):
        self.defender._item = "eviolite"

        self.attacker = helper.add_boosts(
            self.attacker,
            helper.create_stats(0, 3, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender,
            helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Singles' }
            )
        )

        self.assertEqual(sium[0]['0']['damage'], [8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10])

    def test_boost_item_damage_doubles(self):
        self.defender._item = "eviolite"

        self.attacker = helper.add_boosts(
            self.attacker,
            helper.create_stats(0, 3, 0, 0, 0, 0, 0)
        )

        self.defender = helper.add_boosts(
            self.defender,
            helper.create_stats(0, 0, 1, 0, 0, 0, 0)
        )

        sium = json.loads(
            helper.make_request(
                self.attacker, 
                'earthquake', 
                self.defender,
                { 'gameType': 'Doubles' }
            )
        )

        self.assertEqual(sium[0]['0']['damage'], [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7])

if __name__ == '__main__':
    unittest.main()

## QUA
# PokemonHS|dragonair|choiceband|shedskin|tackle,watergun,hiddenpower|Adamant||M||S|84|134,water,,G