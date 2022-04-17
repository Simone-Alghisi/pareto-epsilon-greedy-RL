import { Response, Request, NextFunction } from 'express';
import { CommonMiddleware } from '../common/middlewares/common.middleware';

/**
 * DamageCalcMiddleware class, it extends the {@link CommonMiddleware} class.
 * It aims to manage all the requests received:
 * - In case of errors, the HTTP status code is returned.
 * In this scenarios, it returns only 422
 * - Otherwise the request is allowed to pass
 */
export class DamageCalcMiddleware extends CommonMiddleware {

  /**
   * Function which validates the move field of a request
   * 
   * @param move move field
   * 
   * @returns true if the field is valid
   * @returns false if the field is not valid
  */
  static validateMove(move: any) :boolean{
    return move && DamageCalcMiddleware.validateString(move);
  }

  /**
   * Function which validates the args field of a pokemon in the 
   * request
   * 
   * @param args args field
   * 
   * @returns true if the field is valid
   * @returns false if the field is not valid
  */
  static validateArgs(args: any): boolean {
    return args.species && 
    args.types && 
    args.weightkg && 
    args.level && 
    args.gender && 
    args.ability && 
    args.is_dynamaxed && 
    args.item && 
    args.status && 
    args.toxicCounter && 
    args.curHP && 
    DamageCalcMiddleware.validateString(args.species) &&
    DamageCalcMiddleware.validateStringArray(args.types) &&
    DamageCalcMiddleware.isNumber(args.weightkg) &&
    DamageCalcMiddleware.isNumber(args.level) &&
    DamageCalcMiddleware.stringOrUndefined(args.gender) &&
    DamageCalcMiddleware.stringOrUndefined(args.ability) &&
    DamageCalcMiddleware.validateBoolean(args.is_dynamaxed) &&
    DamageCalcMiddleware.stringOrUndefined(args.item) &&
    DamageCalcMiddleware.stringOrUndefined(args.status) &&
    DamageCalcMiddleware.isNumber(args.toxicCounter) &&
    DamageCalcMiddleware.isNumber(args.curHP)
  }

  /**
   * Function which validates the stats field of a pokemon in the 
   * request
   * 
   * @param stat args field
   * @param boost if it is a boost field
   * 
   * @returns true if the field is valid
   * @returns false if the field is not valid
  */
  static validateStats(stat: any, boost: boolean): boolean{
    let valid: boolean;
    if(boost){
      valid = 
      stat.at && DamageCalcMiddleware.isNumber(stat.at) &&
      stat.df && DamageCalcMiddleware.isNumber(stat.df) &&
      stat.sa && DamageCalcMiddleware.isNumber(stat.sa) &&
      stat.sd && DamageCalcMiddleware.isNumber(stat.sd) &&
      stat.sp && DamageCalcMiddleware.isNumber(stat.sp)
    } else {
      valid =
      (!stat.at || DamageCalcMiddleware.isNumber(stat.at)) &&
      (!stat.df || DamageCalcMiddleware.isNumber(stat.df)) &&
      (!stat.sa || DamageCalcMiddleware.isNumber(stat.sa)) &&
      (!stat.sd || DamageCalcMiddleware.isNumber(stat.sd)) &&
      (!stat.sp || DamageCalcMiddleware.isNumber(stat.sp))
    }
    return valid;
  }

  /**
   * Function which validates the fields of a pokemon in the 
   * request
   * 
   * @param mon pokemon field
   * 
   * @returns true if the field is valid
   * @returns false if the field is not valid
  */
  static validatePokemonFields(mon: any) :boolean{
    return mon && mon.name && mon.args && DamageCalcMiddleware.validateString(mon.name);
  }

  /**
   * Function which validates the batch of requests to the @smogol/calc
   * 
   * @param req Express request
   * @param res Express response 
   * @param next Express next function
   * 
   * In case of errors, it returns error code 422, unprocessable entity
  */
  validateRequestBatch(req: Request, res: Response, next: NextFunction) :void {
    let valid = true;
    // Check if the requests exists and they are an array
    if(req.body.requests && Array.isArray(req.body.requests)){
      // Loop over the requests
      for (const r of req.body.requests) {
        // Loop over the pokemon positions
        for (const pos in r) {
          // Check if the request is valid
          if (
            DamageCalcMiddleware.validatePokemonFields(r[pos].attacker) && 
            DamageCalcMiddleware.validatePokemonFields(r[pos].target) && 
            r[pos].move &&
            DamageCalcMiddleware.validateMove(r[pos].move) && 
            DamageCalcMiddleware.validateArgs(r[pos].attacker.args) && 
            DamageCalcMiddleware.validateArgs(r[pos].target.args) && 
            r[pos].attacker.boost &&
            r[pos].attacker.stats &&
            r[pos].target.sats &&
            r[pos].target.boost &&
            DamageCalcMiddleware.validateStats(r[pos].attacker.boost, true) &&
            DamageCalcMiddleware.validateStats(r[pos].attacker.stats, false) &&
            DamageCalcMiddleware.validateStats(r[pos].target.boost, true) &&
            DamageCalcMiddleware.validateStats(r[pos].target.stats, false)
          ) {
            valid = false;
            break;
          }
        }
      }
    }
    if (valid){
      next()
    } else {
      res.status(422).json({ error: 'Unprocessable entity' });
    }
  }

  /**
   * Function which performs a request setup, which means that
   * it adapts the requests for both posts and gets
   * 
   * @param req Express request
   * @param res Express response 
   * @param next Express next function
   * 
   * In case of errors, it returns error code 422, unprocessable entity
  */
  setupReq(req: Request, res: Response, next: NextFunction) :void{
    if(req.params && !req.body){
      req.body = req.params;
      next();
    } else {
      res.status(422).json({ error: 'Unprocessable entity' });
    }
  }
}