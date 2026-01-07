from enum import Enum
from models.gnns import WSSSAPP02GINE, WSSSAPP02GINEValueNet
from models.mlps import WSSSAPP02MLP, WSSSAPP02MLPValueNet


class ActionNets(Enum):
    WSSSAPP02GINE = WSSSAPP02GINE
    WSSSAPP02MLP = WSSSAPP02MLP


class ValueNets(Enum):
    WSSSAPP02GINEValueNet = WSSSAPP02GINEValueNet
    WSSSAPP02MLPValueNet = WSSSAPP02MLPValueNet