from .lstm import LSTMAM
from .dfsmn import DFSMNAM


def load_model(conf):
    if (conf["model_type"] == "lstm"):
        return LSTMAM(**conf["model_conf"])
    elif (conf["model_type"] == "dfsmn"):
        return DFSMNAM(**conf["model_conf"])
    else:
        print("model type:{%s} is not supported!!!".format(conf["model_type"]))
