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
from pynpm import NPMPackage


def install_pkg(path):
    pkg = NPMPackage("{}/../../package.json".format(path))
    p = pkg.install(wait=False)
    p.wait()


def is_installed() -> bool:
    try:
        # call node to see whether the package is installed
        ps = subprocess.Popen(("npm", "list", "--depth", "0"), stdout=subprocess.PIPE)
        output = subprocess.check_output(
            ("grep", "-c", "@smogon/calc"), stdin=ps.stdout
        )
        ps.wait()
        # decode the bytes read
        output = output.decode("utf-8")
    except subprocess.CalledProcessError as e:
        output = 0
    return bool(output)


def damage_request(parameters=""):
    # holds the directory where python script is located
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not is_installed():
        print("Installing required package...")
        install_pkg(dir_path)
    try:
        # call node function
        output = subprocess.check_output(
            ("npm", "run", "tsc", "{}/ts/dmg_calculator.ts".format(dir_path))
        )
        output = subprocess.check_output(
            ("node", "{}/ts/dmg_calculator.js".format(dir_path))
        )
        # decode the bytes read
        output = output.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        output = None
    return output
