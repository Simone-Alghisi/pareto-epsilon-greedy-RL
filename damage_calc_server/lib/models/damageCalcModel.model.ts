import {Generations, calculate, Pokemon, Move, Field} from '@smogon/calc';

export class Stats {
  hp: number | undefined;
  atk: number | undefined;
  def: number | undefined;
  spa: number | undefined;
  spd: number | undefined;
  spe: number | undefined;

  constructor(
    hp: number | undefined,
    atk: number | undefined,
    def: number | undefined,
    spa: number | undefined,
    spd: number | undefined,
    spe: number | undefined
  ){
    this.hp = hp;
    this.atk = atk;
    this.def = def;
    this.spa = spa;
    this.spd = spd;
    this.spe = spe;
  }

  static describe(): string[] {
    return Object.getOwnPropertyNames(
      new Stats(
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined
      )
    );
  }
}

export class Args {
  species: string | undefined;
  types: string[] | undefined;
  weightkg: number | undefined;
  level: number | undefined;
  gender: string | undefined;
  ability: string | undefined;
  is_dynamaxed: boolean | undefined;
  item: string | undefined;
  status: string | undefined;
  toxicCounter: number | undefined;
  curHP: number | undefined;
  boosts: Stats | undefined;
  stats: Stats | undefined;

  constructor(
    species: string | undefined,
    types: string[] | undefined,
    weightkg: number | undefined,
    level: number | undefined,
    gender: string | undefined,
    ability: string | undefined,
    is_dynamaxed: boolean | undefined,
    item: string | undefined,
    status: string | undefined,
    toxicCounter: number | undefined,
    curHP: number | undefined,
    boosts: Stats | undefined,
    stats: Stats | undefined,
  ){
    this.species = species;
    this.types = types;
    this.weightkg = weightkg;
    this.level = level;
    this.gender = gender;
    this.ability = ability;
    this.is_dynamaxed = is_dynamaxed;
    this.item = item;
    this.status = status;
    this.toxicCounter = toxicCounter;
    this.curHP = curHP;
    this.boosts = boosts;
    this.stats = stats;
  }

  static describe(): string[] {
    return Object.getOwnPropertyNames(
      new Args(
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined
      )
    );
  }

  toString(): string {
    return JSON.stringify(this)
  }

  toObj(): object {
    return JSON.parse(this.toString())
  }
}

export class CalcPokemon {
  name: string | undefined;
  args: Args | undefined;

  constructor(
    name: string | undefined,
    args: Args | undefined
  ){
    this.name = name;
    this.args = args;
  }

  static describe(): string[] {
    return Object.getOwnPropertyNames(
      new CalcPokemon(undefined, undefined)
    );
  }
}

export class CalcField {
  gameType: string | undefined;
  weather: string | undefined;
  terrain: string | undefined;
  isGravity: boolean | undefined;
  
  constructor(
    gameType: string | undefined,
    weather: string | undefined,
    terrain: string | undefined,
    isGravity: boolean | undefined
  ){
    this.gameType = gameType;
    this.weather = weather;
    this.terrain = terrain;
    this.isGravity = isGravity;
  }

  static describe(): string[] {
    return Object.getOwnPropertyNames(
      new CalcField(
        undefined, 
        undefined,
        undefined, 
        undefined
      )
    );
  }

  toString(): string {
    return JSON.stringify(this)
  }

  toObj(): object {
    return JSON.parse(this.toString())
  }
}

/**Class which specifies the schema for a class in the DB */
export class DamageCalcModel {
  /* Variables */
  attacker: CalcPokemon | undefined;
  target: CalcPokemon | undefined;
  move: string | undefined;
  field: CalcField | undefined;

  constructor(
    attacker: CalcPokemon | undefined,
    target: CalcPokemon | undefined,
    move: string | undefined,
    field: CalcField | undefined
  ){
    this.attacker = attacker;
    this.target = target;
    this.move = move;
    this.field = field;
  }

  static describe(): string[] {
    return Object.getOwnPropertyNames(
      new DamageCalcModel(undefined, undefined, undefined, undefined)
    );
  }

  calculate(): any{
    let attacker_name = '';
    let target_name = '';
    let attacker_args: any = {};
    let target_args: any = {};
    const move: string = this.move || '';

    if (this.attacker){
      attacker_name = this.attacker.name || '';
      attacker_args = this.attacker.args || {};
    }
    if (this.target){
      target_name = this.target.name || '';
      target_args = this.target.args || {};
    }

    const gen = Generations.get(8);

    const result: any = calculate(
      gen,
      new Pokemon(gen, attacker_name, attacker_args.toObj()),
      new Pokemon(gen, target_name, target_args.toObj()),
      new Move(gen, move),
      new Field(this.field!.toObj())
    );

    // Add full description, if possible
    try{
      result['description'] = result.fullDesc();
    }catch{
      result['description'] = '';
    }
    return result;
  }
}
