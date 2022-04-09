/***
 * File which deals with batch of requests to '@smogon/calc'
 * 
 * Request format:
 * 
 * {
 *  request: [{
 *       0: {'Attacker': pkmn0, 'Move': m0, 'Defender': d1},
 *       1: {'Attacker': pkmn1, 'Move': m2, 'Defender': d2},
 *       2: {'Attacker': pkmn2, 'Move': m3, 'Defender': d3},
 *       3: {'Attacker': pkmn3, 'Move': m4, 'Defender': d4}
 *     }
 *   ..
 *  ]
 * }
 * 
 * Response format:
 * 
 * {
 *  results: [{
 *       0: { 'damage': ...},
 *       1: { 'damage': ...},
 *       2: { 'damage': ...},
 *       3: { 'damage': ...}
 *     }
 *   ..
 *  ]
 * }
 */
import {Generations, calculate, Pokemon, Move} from '@smogon/calc';

// Variables
const PROPERTIES: Array<string> = ['Attacker', 'Defender', 'Move']; // Properties of each request
const REQLIST_ATTRIBUTE: string = 'requests';
const RESULTS_ATTRIBUTE: string = 'results';
const EL_PER_REQUEST: number = 4
let request: string = process.argv[2]; // Request
let requestJson: any = null; // Request JSON
let answer: any = {}; // Answer to be returned
answer[RESULTS_ATTRIBUTE] = [];

/**
 * Check if the request fulfills the properties listed
 * in the PROPERTIES array
 * 
 * Args:
 *  request: object
 */
function isValidRequest(request: object) {
  // Check field
  for(let i = 0; i < PROPERTIES.length; i++) {
    if (!request.hasOwnProperty(PROPERTIES[i])) {
      throw "Missing property " + PROPERTIES[i] + " in the damage request " + JSON.stringify(request);
    }
  }
}

/**
 * Asks the damage calculator for the damage
 * 
 * Args:
 *  request: any
 * 
 * returns resultArray any
 */
function damageRequest(request: any){
  let requestArray: any = {}
  // Dealing batches of EL_PER_REQUEST elements
  for(let i = 0; i < EL_PER_REQUEST; i++){
    isValidRequest(request[i]);  // Valid request
    const gen = Generations.get(8);
    let result: any = calculate(
      gen,
      new Pokemon(gen, request[i].Attacker),
      new Pokemon(gen, request[i].Defender),
      new Move(gen, request[i].Move),
      undefined
    );
    // Add full description, if possible
    try{
      result['description'] = result.fullDesc();
    }catch{
      result['description'] = '';
    }
    requestArray[i] = result;
  }
  return requestArray;
}

// The request should contain the JSON object
if(process.argv.length < 3){
  throw "Empty request";
}

// Parse request
try{
  requestJson = JSON.parse(request); // Parse the JSON
} catch(e){
  throw "Invalid JSON";
}

// Check if the list is present
if(!requestJson.hasOwnProperty(REQLIST_ATTRIBUTE)){
  throw "Missing property " + REQLIST_ATTRIBUTE;
}

// Loop over the requests and pushes them in the array
requestJson[REQLIST_ATTRIBUTE].forEach(function (value: any) { 
  answer[RESULTS_ATTRIBUTE].push(damageRequest(value));
});

// Output the result
console.log(JSON.stringify(answer));