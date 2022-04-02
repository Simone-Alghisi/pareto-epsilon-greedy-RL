"use strict";
exports.__esModule = true;
var calc_1 = require("@smogon/calc");
var gen = calc_1.Generations.get(1);
var result = (0, calc_1.calculate)(gen, new calc_1.Pokemon(gen, 'Gengar'), new calc_1.Pokemon(gen, 'Vulpix'), new calc_1.Move(gen, 'Surf'), new calc_1.Field({ defenderSide: { isLightScreen: true } }));
console.log(result);
