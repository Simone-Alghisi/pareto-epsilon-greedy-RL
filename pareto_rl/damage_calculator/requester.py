""" requester.py
Module which allows to perform a request to the damage calculator API

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.

Authors:

- Simone Alghisi (simone.alghisi-1@studenti.unitn.it)
- Samuele Bortolotti (samuele.bortolotti@studenti.unitn.it)
- Massimo Rizzoli (massimo.rizzoli@studenti.unitn.it)
- Erich Robbi (erich.robbi@studenti.unitn.it)
"""

import subprocess
import os
import logging


def is_transpiled(file_path: str) -> bool:
    r"""Returns true if the TypeScript file has been transpiled

    Args:
        file_path [str]: path of the transpiled JavaScript file
    """
    return os.path.isfile(file_path)


def damage_request(parameters: str = ""):
    r"""Asks the @smogol/calc Node module to compute the damage

    Args:
        parameters [str]: parameters of the @smogol/calc library
    """
    # logger
    logger = logging.getLogger(__name__)
    logger.info(parameters)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not is_transpiled("{}/ts/dmg_calculator.js".format(dir_path)):
        # call transpile function
        output = subprocess.check_output(
            ("npm", "run", "tsc", "{}/ts/dmg_calculator.ts".format(dir_path))
        )
        logger.info("TypeScript dmg_calculator.ts transpiled")
    try:
        # call node function
        output = subprocess.check_output(
            ("node", "{}/ts/dmg_calculator.js".format(dir_path), "{}".format(parameters))
        )
        # decode the bytes read
        output = output.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        output = None
    return output
