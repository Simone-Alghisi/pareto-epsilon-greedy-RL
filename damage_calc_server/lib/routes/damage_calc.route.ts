import { Application } from 'express';
import { CommonRoutes } from '../common/routes/common.routes'
import { ConfigureRoutes } from '../common/interfaces/configureRoutes.interface'
import { DamageCalcController } from  '../controllers/damage_calc.controller'
import { DamageCalcMiddleware } from '../middlewares/damage_calc.middleware'

/**
 * DamageCalcRoute class, it extends the {@link CommonRoutes} class and implements the {@link ConfigureRoutes} interface.
 * It aims to manage all the requests received for the resource _/damagecalc_.
 * It sets the middlewares and the methods that should be called for a specific operation
 */
export class DamageCalcRoute extends CommonRoutes implements ConfigureRoutes {

  /**
   * Constructor that calls the consutructor of CommonRoutes and calls the method that define all the routes
   * 
   * @param app instance of the node.js server
   */
  constructor(app: Application){
    super(app, 'DamageCalc');
    this.configureRoutes();
  }

  /**
   * Configures the route for some HTTP methods
   */
  configureRoutes(): void {
    /** Instance of damage calc controller which implements the business logic*/
    const damageController: DamageCalcController = new DamageCalcController();
    /** Instance of user middleware which checks every request on user resources*/
    const damageCalcMiddleware: DamageCalcMiddleware = new DamageCalcMiddleware();

    /**
     * Route for the get method on the damagecalc resource.
     * The request is routed though:
     * - setup req
     * - validateRequest batch
     * as it expect to have a batch of requests.
     * The output will be the damage returned by @smogon/calc for each reques
    */
    this.app.get('/api/v1/damagecalc', [
      damageCalcMiddleware.setupReq,
      damageCalcMiddleware.validateRequestBatch,
      damageController.calc
    ]);

    /** 
     * Route for the post method on the damagecalc resources 
     * The request is routed though:
     * - setup req
     * - validateRequest batch
     * as it expect to have a batch of requests.
     * The output will be the damage returned by @smogon/calc for each reques
    */
    this.app.post('/api/v1/damagecalc', [
      damageCalcMiddleware.setupReq,
      damageCalcMiddleware.validateRequestBatch,
      damageController.calc
    ]);
  }
}
