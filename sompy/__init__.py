from logging.config import dictConfig
import matplotlib
import sys
sys.path.append(r"C:\DevSource\Shuyang-GEOG676\Projects\sompy")

dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "root": {
        "level": "NOTSET",
        "handlers": ["console"]
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "basic"
        }
    },
    "formatters": {
        "basic": {
            "format": '%(message)s'
        }
    }
})



from sompy import SOMFactory
from visualization import *