import json


def create_readmefile(params):
    LINES = [
        "README\n",
        "==========\n",
        "MESSAGE:\n" + str(params["message"]) + "\n",
        "==========\n",
        json.dumps(params, indent=4)
    ]

    with open(params["output"] + "/README.txt", "w") as f:
        f.writelines(LINES)
