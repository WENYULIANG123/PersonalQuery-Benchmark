from .amazon import AmazonSKB
try:
    from .prime import PrimeSKB
except (ImportError, ModuleNotFoundError):
    PrimeSKB = None
try:
    from .mag import MagSKB
except (ImportError, ModuleNotFoundError):
    MagSKB = None
from .knowledge_base import SKB

REGISTERED_SKBS = [
    'amazon', 
    'prime', 
    'mag'
]
