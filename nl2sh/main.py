import os

from nl2sh.agents.clarifier import Clarifier
from nl2sh.agents.composer import Composer
from nl2sh.agents.inspector import Inspector


# States
INIT = 'init'
CLARIFIED = 'clarified'
COMPOSED = 'composed'
NOT_PASS = 'not_passed'
DONE = 'done'

clarifier = Clarifier()
inspector = Inspector()
composer = Composer()

# state machine
Sched = {
    INIT: clarifier,
    CLARIFIED: composer,
    COMPOSED: inspector,
    NOT_PASS: composer,
}


class Pipeline:
    pass


def eval():
    pass


def try_it():
    pass