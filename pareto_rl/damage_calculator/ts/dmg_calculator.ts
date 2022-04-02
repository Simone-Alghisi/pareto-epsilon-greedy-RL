import {Generations, calculate, Pokemon, Move, Field} from '@smogon/calc';

const gen = Generations.get(1);
const result = calculate(
  gen,
  new Pokemon(gen, 'Gengar'),
  new Pokemon(gen, 'Vulpix'),
  new Move(gen, 'Surf'),
  new Field({defenderSide: {isLightScreen: true}})
);

console.log(result)