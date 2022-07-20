import pareto_rl.damage_calculator.requester as rq
from poke_env.environment.pokemon import Pokemon 
from pareto_rl.dql_agent.utils.utils import prepare_pokemon_request
from poke_env.utils import to_id_str, compute_raw_stats
from typing import List

# =========== DUMMY MONS ===========
"""
Bulbasaur @ Eviolite
 Level: 5
 Calm Nature
 Ability: Overgrow
 EVs: 252 Atk / 236 SpD / 252 Spe
 - Giga Drain
 - Leech Seed
 - Sludge Bomb
 - Toxic
"""

"""
Wingull @ Life Orb
Level: 5
Timid Nature
Ability: Hydration
EVs: 36 HP / 236 SpA / 236 Spe
- Scald
- Hurricane
- U-turn
- Knock Off
"""

'''
export declare class Field implements State.Field {
    gameType: GameType;
    weather?: Weather;
    terrain?: Terrain;
    isGravity: boolean;
    attackerSide: Side;
    defenderSide: Side;
    constructor(field?: Partial<State.Field>);
    hasWeather(...weathers: Weather[]): boolean;
    hasTerrain(...terrains: Terrain[]): boolean;
    swap(): this;
    clone(): Field;
}
'''

'''
export declare type AbilityName = string & As<'AbilityName'>;
export declare type ItemName = string & As<'ItemName'>;
export declare type MoveName = string & As<'MoveName'>;
export declare type SpeciesName = string & As<'SpeciesName'>;
export declare type StatusName = 'slp' | 'psn' | 'brn' | 'frz' | 'par' | 'tox';
export declare type GameType = 'Singles' | 'Doubles';
export declare type Terrain = 'Electric' | 'Grassy' | 'Psychic' | 'Misty';
export declare type Weather = 'Sand' | 'Sun' | 'Rain' | 'Hail' | 'Harsh Sunshine' | 'Heavy Rain' | 'Strong Winds';
export declare type NatureName = 'Adamant' | 'Bashful' | 'Bold' | 'Brave' | 'Calm' | 'Careful' | 'Docile' | 'Gentle' | 'Hardy' | 'Hasty' | 'Impish' | 'Jolly' | 'Lax' | 'Lonely' | 'Mild' | 'Modest' | 'Naive' | 'Naughty' | 'Quiet' | 'Quirky' | 'Rash' | 'Relaxed' | 'Sassy' | 'Serious' | 'Timid';
export declare type TypeName = 'Normal' | 'Fighting' | 'Flying' | 'Poison' | 'Ground' | 'Rock' | 'Bug' | 'Ghost' | 'Steel' | 'Fire' | 'Water' | 'Grass' | 'Electric' | 'Psychic' | 'Ice' | 'Dragon' | 'Dark' | 'Fairy' | '???';
export declare type MoveCategory = 'Physical' | 'Special' | 'Status';
export declare type MoveTarget = 'adjacentAlly' | 'adjacentAllyOrSelf' | 'adjacentFoe' | 'all' | 'allAdjacent' | 'allAdjacentFoes' | 'allies' | 'allySide' | 'allyTeam' | 'any' | 'foeSide' | 'normal' | 'randomNormal' | 'scripted' | 'self';
'''

def create_stats(
  accuracy: int,
  atk: int,
  defence: int,
  evasion: int,
  spa: int,
  spd: int,
  spe: int
) -> dict:
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
  Add boosts
  """
  mon._boosts = boosts
  return mon

def clear_boosts(mon: Pokemon):
  """
  Add boosts
  """
  return add_boosts(
    mon,
    create_stats(0, 0, 0, 0, 0, 0, 0)
  )

def compute_stats(
  mon: Pokemon,
  ivs: List[int],
  evs: List[int],
  nature: str
):
  raw = compute_raw_stats(
    mon.species,
    ivs,
    evs,
    mon._level,
    nature
  )
  return {
    "atk": raw[0], 
    "def": raw[1], 
    "spa": raw[2], 
    "spd": raw[3], 
    "spe": raw[4]
  }

def make_request(
  attacker: Pokemon, 
  move: str,
  defender: Pokemon,
  field: dict
) -> str:
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
      "field": field
    },
  }
  # prepare the request
  request = {'requests': [ singleton_request ]}
  # return after sending the request and await
  return rq.damage_request_server(request)