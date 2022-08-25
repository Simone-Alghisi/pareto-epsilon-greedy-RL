# FOLDERS
VENV := venv
PROJECT_NAME := pareto
DAMAGE_CALC_FOLDER := damage_calc_server

# PROGRAMS AND FLAGS
PYTHON := python3
PYFLAGS := -m
MAIN := pareto_rl
MAIN_FLAGS :=
PIP := pip

# ======= TRAIN =========
TRAIN := rlagent
TRAIN_FLAGS :=

# ======= FORMAT ========
FORMAT := black
FORMAT_FLAG := pareto_rl

# ======= TEST  =========
TEST := rlagent
RUN_NUMBER := 581
TEST_FLAGS := --test $(RUN_NUMBER) --fc reward

# ======= UNIT TEST  ====
UNITTEST := unittest
UNITTEST_FLAGS := discover -s test

# ======= PARETO  =========
PARETO := pareto
PARETO_FLAG :=

# ======= PARETO BATTLE =========
PARETO_BATTLE := pareto-battle
PARETO_BATTLE_FLAG := --player ParetePareteParete

# ======= DOC   =========
AUTHORS := --author "Simone Alghisi, Samuele Bortolotti, Massimo Rizzoli, Erich Robbi"
VERSION :=-r 0.1
LANGUAGE := --language en
SPHINX_EXTENSIONS := --extensions sphinx.ext.autodoc --extensions sphinx.ext.napoleon --extensions sphinx.ext.viewcode --extensions myst_parser
DOC_FOLDER := docs

## Quickstart
SPHINX_QUICKSTART := sphinx-quickstart
SPHINX_QUICKSTART_FLAGS := --sep --no-batchfile --project ParetoRL $(AUTHORS) $(VERSION) $(LANGUAGE) $(SPHINX_EXTENSIONS)

# Build
BUILDER := html
SPHINX_BUILD := make $(BUILDER)
SPHINX_API_DOC := sphinx-apidoc
SPHINX_API_DOC_FLAGS := -o $(DOC_FOLDER)/source .
SPHINX_THEME = sphinx_rtd_theme
DOC_INDEX := index.html

# INDEX.rst
define INDEX

.. pareto documentation master file, created by
   sphinx-quickstart on Fri Apr. 1 10:51:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.md
	 :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 5
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

endef

export INDEX

# COLORS
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NONE := \033[0m

# COMMANDS
ECHO := echo -e
MKDIR := mkdir -p
OPEN := xdg-open
SED := sed
GIT := git
CD := cd
NPM := npm
CP := cp

# RULES
.PHONY: help env install install-dev install-showdown train test pareto doc doc-layout format start-showdown pareto-battle start-damage-calc-server install-damage-calc-server unittest

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* env 				: generates the virtual environment using venv\n \
	* install			: install the requirements listed in requirements.txt\n \
	* install-dev			: install the development requirements listed in requirements.dev.txt\n \
	* install-showdown		: install the pokémon showdown server\n \
	* install-damage-calc-server	: install the damage calculator server\n \
	* doc-layout 			: generates the Sphinx documentation layout\n \
	* format 			: format the code using black\n \
	* doc 				: generates the documentation (requires an existing documentation layout)\n \
	* open-doc 			: opens the documentation\n \
	* train 			: train the agent against MaxDamagePlayer\n \
	* test 			: test the agent against MaxDamagePlayer\n \
	* start-showdown 		: starts the showdown server\n \
	* start-damage-calc-server 	: starts the damage calculator server\n \
	* pareto-battle 		: starts a battle with an agents having Pareto optimal moves"

env:
	@$(ECHO) '$(GREEN)Creating the virtual environment..$(NONE)'
	@$(MKDIR) $(VENV)
	@$(eval PYTHON_VERSION=$(shell $(PYTHON) --version | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' | cut -f1,2 -d'.'))
	@$(PYTHON_VERSION) -m venv $(VENV)/$(PROJECT_NAME)
	@$(ECHO) '$(GREEN)Done$(NONE)'

install:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PIP) install -r requirements.txt
	@$(NPM) install
	@$(ECHO) '$(GREEN)Done$(NONE)'

install-dev:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PIP) install -r requirements.dev.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

install-showdown:
	@$(ECHO) '$(GREEN)Installing Pokémon Showdown server..$(NONE)'
	@$(GIT) clone https://github.com/smogon/pokemon-showdown.git
	@$(CD) pokemon-showdown; \
	 $(NPM) install; \
	 $(CP) config/config-example.js config/config.js; \
	 $(SED) -i 's/exports.repl = true/exports.repl = false/g' config/config.js; \
	 $(SED) -i 's/exports.noguestsecurity = false/exports.noguestsecurity = true/g' config/config.js
	@$(ECHO) '$(GREEN)Done$(NONE)'

start-showdown:
	@$(ECHO) '$(BLUE)Starting Pokémon Showdown server..$(NONE)'
	(sleep 1; $(OPEN) http://localhost:8000) &
	@$(CD) pokemon-showdown; \
	$(NPM) start --no-secure
	@$(ECHO) '$(BLUE)Done$(NONE)'

install-damage-calc-server:
	@$(ECHO) '$(GREEN)Installing Damage Calc server..$(NONE)'
	@$(CD) $(DAMAGE_CALC_FOLDER); \
	 $(NPM) install;
	@$(ECHO) '$(GREEN)Done$(NONE)'

start-damage-calc-server:
	@$(ECHO) '$(BLUE)Starting Damage Calc server..$(NONE)'
	@$(CD) $(DAMAGE_CALC_FOLDER); \
	$(NPM) run build; \
	$(NPM) run start;
	@$(ECHO) '$(BLUE)Done$(NONE)'

doc-layout:
	@$(ECHO) '$(BLUE)Generating the Sphinx layout..$(NONE)'
	$(SPHINX_QUICKSTART) $(DOC_FOLDER) $(SPHINX_QUICKSTART_FLAGS)
	@$(ECHO) "\nimport os\nimport sys\nsys.path.insert(0, os.path.abspath('../..'))" >> $(DOC_FOLDER)/source/conf.py
	@$(ECHO) "$$INDEX" > $(DOC_FOLDER)/source/index.rst
	@$(SED) -i -e "s/html_theme = 'alabaster'/html_theme = '$(SPHINX_THEME)'/g" $(DOC_FOLDER)/source/conf.py
	@$(ECHO) '$(BLUE)Done$(NONE)'

doc:
	@$(ECHO) '$(BLUE)Generating the documentation..$(NONE)'
	$(SPHINX_API_DOC) $(SPHINX_API_DOC_FLAGS)
	cd $(DOC_FOLDER); $(SPHINX_BUILD)
	@$(ECHO) '$(BLUE)Done$(NONE)'

open-doc:
	@$(ECHO) '$(BLUE)Open documentation..$(NONE)'
	$(OPEN) $(DOC_FOLDER)/build/$(BUILDER)/$(DOC_INDEX)
	@$(ECHO) '$(BLUE)Done$(NONE)'

train:
	@$(ECHO) '$(BLUE)Training the network..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(TRAIN) $(TRAIN_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

test:
	@$(ECHO) '$(BLUE)Testing the network..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(TEST) $(TEST_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

unittest:
	@$(ECHO) '$(BLUE)Running the unittests..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(UNITTEST) $(UNITTEST_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

pareto-battle:
	@$(ECHO) '$(BLUE)Battle with Pareto moves..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(MAIN) $(MAIN_FLAGS) $(PARETO_BATTLE) $(PARETO_BATTLE_FLAG)
	@$(ECHO) '$(BLUE)Done$(NONE)'

format:
	@$(ECHO) '$(BLUE)Formatting the code..$(NONE)'
	@$(FORMAT) $(FORMAT_FLAG)
	@$(ECHO) '$(BLUE)Done$(NONE)'