import {Generations, calculate, Pokemon, Move, Field} from '@smogon/calc';

if(process.argv.length < 3){
  throw "Empty request";
}

let properties: Array<string> = ['Attacker', 'Defender', 'Move'];
let request: string = process.argv[2];
let request_json = null;

// Parse request
try{
  request_json = JSON.parse(request);
} catch(e){
  throw "Invalid JSON"
}

// Check field
for(let i = 0; i < properties.length; i++) {
  if (!request_json.hasOwnProperty(properties[i])) {
    throw "Missing property " + properties[i] + " in the damage request";
  }
} 

const gen = Generations.get(8);
const result = calculate(
  gen,
  new Pokemon(gen, request_json.Attacker),
  new Pokemon(gen, request_json.Defender),
  new Move(gen, request_json.Move),
  undefined //new Field({defenderSide: {isLightScreen: true}})
);

// add full description
try{
  result['description'] = result.fullDesc();
}catch{
  result['description'] = '';
}
console.log(JSON.stringify(result))