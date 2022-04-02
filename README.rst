Pareto Epsilon Greedy RL
========================

Repository for the I2BIAI project.

About
-----

* Authors  :

  - `Alghisi Simone <https://github.com/Simone-Alghisi>`_\
  - `Bortolotti Samuele <https://github.com/samuelebortolotti>`_\ 
  - `Rizzoli Massimo <https://github.com/massimo-rizzoli>`_\
  - `Erich Robbi <https://github.com/erich-r>`_\

* Licence   : GPL v3+


General information
-------------------

In order to dive deeper into the world, some base knowledge needs to be
comprehanded. In particular, it is fundamental to understand how damage
calculation works, because it will be (very likely) the core of both
NSGA-II and RL.

Damage calculation
~~~~~~~~~~~~~~~~~~

More detailed info can be found
`here <https://bulbapedia.bulbagarden.net/wiki/Damage>`__, I will cover
the most important parts starting from the formula, by gradually adding
information whenever is needed in order to complicate things.

.. math::

   Damage = \left( \frac{\left( \frac{2 \times Level}{5} + 2 \right) \times Power \times 
   A/D}{50} + 2 \right) \times Targets \times Weather \times Critical \times random \times 
   STAB \times Type \times Burn \times other

Let us now tackle the most important parts:

-  Level is as the name suggests the level of the *attacking pokemon*
-  Power is the *base power* of the move that the attacking pokemon is
   going to use on the defending pokemon
-  A is the *effective attack* of the *attacking pokemon*. HOWEVER, keep
   in mind that

   1. this includes both boosts and debuffs (do not worry, showdown
      always show them)
   2. there are 2 kinds of attacks: physical and special. This means
      that whenever we are calculating the damage of a move, we need
      first of all to understand if the move is physical or special, to
      consider the correct statistic.
   3. if the pokemon crits… all debuffs of the attacker are ignored
      (also buffs of the defender…)

-  D is the *effective defense* of the *defending pokemon*. Of course,
   we have the same considerations that we did before. Furthermore, we
   also have a bit of unknown to tackle regarding that we won’t exactly
   know the correct value of the defending one (only the game will know
   but we could solve possibly some sort of equation to get an
   approximation)
-  Targets is a multiplier based on the number of targets (generally 1)
-  Weather is the weather condition, which could boosts or debuffs some
   moves, e.g.

   1. *sun* boosts fire type moves of 1.5 while debuffs water type moves
      of 0.5
   2. *harsh sunlight* does the same… however competely nullifies water
      type moves

-  Critical, if the pokemon perform a critical hit, the damage is
   multipled by 1.5
-  random is a random integer percentage between 85% and 100%
   (inclusive)
-  STAB is a 1.5 multiplier which comes from the fact that the *attacker
   pokemon* uses a move with the same type of the pokemon
-  Type, which I would have called type resistance, is a multiplier
   based in the effectiveness of the move used on the *defender
   pokemon*, e.g. 

   1. fire is super-effective on grass, so it’s 1.5
   2. fire is not very effective on water, so it’s 0.5

-  Burn, is a status condition which halves ONLY physical damage if
   present
-  other, you do not want to know (however just think about that there
   are objects and abilities in this game too…)

How to proceed
^^^^^^^^^^^^^^

I think that some of the previous multipliers are quite difficult to
implement for the moment, so I would like to tackle them in a certain
way in order to implement things gradually.

However, using a damage calculator directly could also take into account
other stuff so the best idea is to start with this and then trasnsition
to smogon damage calculator (even if it works with npm)

First steps
'''''''''''

I think that the most painless implementation requires us to deal with
the following information:

-  Level
-  Power
-  A
-  D (for the moment without worrying about the fact that it could not
   be fixed)
-  random
-  Critical
-  STAB
-  Type (which requires a type effectiveness table of some kind)

Range
'''''

First of all it must be understood that damages are always between a
range: there is no such a thing as a single number because, even if D is
completely known, there are always Critical and also random which can
increase/decrease the damage.

For this reason, we need a way to express the range to the net and adapt
it while we understand more things about our opponent.

Expected damage
'''''''''''''''

Given that some moves deal lots of damage with high risk, I would like
to insert in the damage calculation also the expected value of the
damage. In particular,

.. math:: ADamage = Damage \times Accuracy

where Accuracy is the move *effective accuracy* which depends on 1. the
accuracy of the *attacking pokemon* 2. the evasiness of the *defending
pokemon*

At the end, this is another thing to take into account and most damage
calculators do not actually care about it. However, I would say that is
fundamental for Utilities.

In which order release contraints?
''''''''''''''''''''''''''''''''''

Very interesting question… I actually do not know myself because we
could either go for something which seems easy and then regret that
choice because it’s actually not that easy.

For example, there are only 4 weather conditions in this game + 3 very
special ones which we could easily remove from the equation. In
particular, - sun boosts fire type moves and decreases water type moves
- rain… does exactly the opposite - sandstorm boosts the special defense
of rock type pokemon AND causes damage every turn to some pokemon with
certain types - hail does damage at every turn to all non-ice pokemon

MOREOVER, some of them also influence the accuracy of some moves… At the
end, it becomes a mess because even if it’s not a problem for the final
damage, it becomes a problem for the fact that we will hit or miss a
certain move.

Given that, each of the constraint can be released while keeping some
other constraints. I would infact say that the best way to go is
(provided that we move from 2 to 3/4 pokemon)

-  equation for updating the current D value
-  Burn
-  Weather (without caring about accuracy)
-  Targets
-  other

What about showdown
'''''''''''''''''''

Once again however, everything could become quite straightforward from
showdown because it gives us additional information about everything,
such as

-  *effective attack*
-  *effective accuracy*
-  boosts and debuffs
-  weather conditions (and their duration)
-  something which I’m surely missing but could be useful

At the end, depending on what we are using showdown will tell us a lot.

Usage
=====

To facilitate the use of the application, a ``Makefile`` has been provided; to see its functions, simply call the appropriate ``help`` command with `GNU/Make <https://www.gnu.org/software/make/>`_

.. code-block:: shell

   make help

0. Set up
---------

For the development phase, the Makefile provides an automatic method to create a virtual environment.

If you want a virtual environment for the project, you can run the following commands:

.. code-block:: shell

   pip install --upgrade pip

Virtual environment creation in the venv folder

.. code-block:: shell

   make env

Virtual environment activation

.. code-block:: shell

   source ./venv/pareto/bin/activate

Install the requirements listed in ``requirements.txt``

.. code-block:: shell

   make install

1. Documentation
----------------

The documentation is built using `Sphinx v4.3.0 <https://www.sphinx-doc.org/en/master/>`_.

If you want to build the documentation, you need to enter the project folder first:

.. code-block:: shell

   cd pareto_rl

Install the development dependencies [``requirements.dev.txt``]

.. code-block:: shell

   make install-dev

Build the Sphinx layout

.. code-block:: shell

   make doc-layout

Build the documentation

.. code-block:: shell

   make doc

Open the documentation

.. code-block:: shell

   make open-doc

2. Pareto front
---------------

To run the Pareto front you can either type:

.. code-block:: shell

   python -m pareto_rl pareto

Or employ the command of the GNU/Makefile

.. code-block:: shell

   make pareto

3. Training
-----------

Train a model
~~~~~~~~~~~~~

4. Testing
----------
