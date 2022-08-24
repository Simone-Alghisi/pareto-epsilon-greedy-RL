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
import requests

#: node server url, in the case it is employed
SERVER_URL = "http://localhost:8080/api/v1/damagecalc"


def is_transpiled(file_path: str) -> bool:
    r"""Returns true if the TypeScript file has been transpiled

    Args:
        file_path [str]: path of the transpiled JavaScript file
    """
    return os.path.isfile(file_path)


def damage_request_server(request: dict):
    r"""Asks the @smogol/calc Node module to compute the damage

    The @smogol/calc should be hosted on a node server available at
    SERVER_URL url

    Args:
        request [dict]: dictionary request
    """
    result = requests.post(SERVER_URL, json=request)
    if result.status_code != 200:
        raise Exception(
            f"Error in the request format, server answered with status code {result.status_code}"
        )
    return result.text


def damage_request_subprocess(parameters: str = ""):
    r"""Asks the @smogol/calc Node module to compute the damage

    Args:
        parameters [str]: parameters of the @smogol/calc library
    """
    # logger
    logger = logging.getLogger(__name__)
    logger.debug(parameters)
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
            (
                "node",
                "{}/ts/dmg_calculator.js".format(dir_path),
                "{}".format(parameters),
            )
        )
        # decode the bytes read
        output = output.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        output = None
    return output
