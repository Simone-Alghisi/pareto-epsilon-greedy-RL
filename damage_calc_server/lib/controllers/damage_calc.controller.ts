import { Request, Response } from 'express';
import { DamageCalcModel, CalcPokemon, Args, Stats, CalcField } from '../models/damageCalcModel.model';

/**
 * DamageController class
 * It aims to manage all the operations that involves the damage calculator.
 */
export class DamageCalcController{

  /**
   * Constructor
   */
  constructor() {}

  /**
   * Function which answers the api request with the json 
   * returned by the damage calculator request
   * 
   * @param req Express request
   * @param res Express response 
   * 
   * It returns error code 200, success
  */
  calc(req: Request, res: Response): void {
    // Json array
    const json_array: any[] = [];
    // if (field) 
    // field = new CalcField();
    for (const r of req.body.requests) {
      // Answer of each request
      const answer: any = {};
      // Loop over the pokemon positions
      for(const pos in r){ 
        const field = r[pos].field;
        // Instantiate the Damage Calc
        const damage_calc_model = new DamageCalcModel(
          new CalcPokemon(r[pos].attacker.name, new Args(      
            r[pos].attacker.species,
            r[pos].attacker.types,
            r[pos].attacker.weightkg,
            r[pos].attacker.level,
            r[pos].attacker.gender,
            r[pos].attacker.ability,
            r[pos].attacker.is_dynamaxed,
            r[pos].attacker.item,
            r[pos].attacker.status,
            r[pos].attacker.toxicCounter,
            new Stats(
              r[pos].attacker.boosts.hp,
              r[pos].attacker.boosts.at,
              r[pos].attacker.boosts.df,
              r[pos].attacker.boosts.sa,
              r[pos].attacker.boosts.sd,
              r[pos].attacker.boosts.sp,
            ),
            new Stats(
              r[pos].attacker.stats.hp,
              r[pos].attacker.stats.at,
              r[pos].attacker.stats.df,
              r[pos].attacker.stats.sa,
              r[pos].attacker.stats.sd,
              r[pos].attacker.stats.sp,
            ),
          )),
          new CalcPokemon(r[pos].target.name, new Args(
            r[pos].target.species,
            r[pos].target.types,
            r[pos].target.weightkg,
            r[pos].target.level,
            r[pos].target.gender,
            r[pos].target.ability,
            r[pos].target.is_dynamaxed,
            r[pos].target.item,
            r[pos].target.status,
            r[pos].target.toxicCounter,
            new Stats(
              r[pos].target.boosts.hp,
              r[pos].target.boosts.at,
              r[pos].target.boosts.df,
              r[pos].target.boosts.sa,
              r[pos].target.boosts.sd,
              r[pos].target.boosts.sp,
            ),
            new Stats(
              r[pos].target.stats.hp,
              r[pos].target.stats.at,
              r[pos].target.stats.df,
              r[pos].target.stats.sa,
              r[pos].target.stats.sd,
              r[pos].target.stats.sp,
            ),
          )),
          r[pos].move,
          new CalcField(
            field.gameType,
            field.weather,
            field.terrain,
            field.isGravity
          )
        )
        // Add the calculation to the answer
        answer[pos] = damage_calc_model.calculate();
      }
      // Push the request anwer to the array
      json_array.push(answer);
    }
    // Return the array
    res.status(200).send(JSON.stringify(json_array))
  }
}