/***
 * File which deals with a batch of requests to '@smogon/calc'
 * 
 * Request format:
 * 
 * [
*        {
*          -1: 
*            {
*                'attacker': 
*                      {
*                         'name': 'mon_name', 
*                         'args': {...}
*                      }
*                , 
*                'target': 
*                      {
*                         'name': 
*                         'mon_name', 
*                         'args': {...}
*                      }, 
*                'move': 'move_name', 
*                'field': {...}
*            }
*        }       
*        ...
 * ]
 * 
 * Response format:
 * 
 * [
*        {
*          -1: 
*            {
*              hp: ...
*              damage: ---
*            }
*        }       
*        ...
 *  ]
 */
import {Generations, calculate, Pokemon, Move} from '@smogon/calc';

// Variables
const PROPERTIES: Array<string> = ['attacker', 'target', 'move', 'field']; // Properties of each request
let request: string = process.argv[2]; // Request
let requestJson: any = null; // Request JSON
let answer: any = []; // Answer to be returned

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
    if (!Object.prototype.hasOwnProperty.call(request, PROPERTIES[i])){
      throw "Missing property " + PROPERTIES[i] + " in the damage request " + JSON.stringify(request);
    }
  }
}

/**
 * Check if the dictionary is not empty
 * If that is the case then return the actual value
 * Otherwise return undefined
 * 
 * Args:
 *  param: object
 * 
 * Returns:
 *  param: object or undefined
 */
function getOptionalParam(param: object){
  let field = undefined; // field
  if (Object.keys(param).length != 0){ // non empty field
    field = param;
  }
  return field;
}

/**
 * Asks the damage calculator for the damage
 * 
 * Args:
 *  request: any
 * 
 * returns requestDict: any
 */
function damageRequest(request: any){
  let requestDict: any = {}
  // Loop over the object attributes
  for (let pos in request) {
    // Get the requester value
    let value = request[pos];
    isValidRequest(value);  // Valid request
    const gen = Generations.get(8);
    let field = getOptionalParam(value['field']);
    // Get args
    let args_attacker = getOptionalParam(value['attacker']['args']); // attacker args
    let args_target = getOptionalParam(value['target']['args']) // target name
    let result: any = calculate(
      gen,
      new Pokemon(gen, value['attacker']['name'], args_attacker),
      new Pokemon(gen, value['target']['name'], args_target),
      new Move(gen, value['move']),
      field
    );
    // Add full description, if possible
    try{
      result['description'] = result.fullDesc();
    }catch{
      result['description'] = '';
    }
    requestDict[pos] = result
  }
  return requestDict;
}

// The request should contain the JSON object
if(process.argv.length < 3){
  throw "Empty request";
}

// Parse request
try{
  requestJson = JSON.parse(request);
} catch(e){
  throw "Invalid JSON";
}

// Loop over the requests and push them in the array
requestJson.forEach(function (req: any) { 
  answer.push(damageRequest(req));
});

// Output the result
console.log(JSON.stringify(answer));