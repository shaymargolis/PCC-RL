def create_readmefile(comb_kwargs, twop_kwargs, output):
    f = open(output + "/README.txt", "w")

    LINES = [
        "Parameters\n",
        "==========\n",
        "COMB - \n",
        "\tLR        = %d\n" % comb_kwargs['lr'],
        "\tLOWER_LR  = %d\n" % comb_kwargs['lower_lr'],
        "\tMIN_PROBA = %.3f\n" % comb_kwargs['min_proba_thresh'],
        "TWOP - \n",
        "\tLR        = %d\n" % twop_kwargs['lr'],
        "\tLOWER_LR  = %d\n" % twop_kwargs['lower_lr'],
        "\tDELTA     = %.3f\n" % twop_kwargs['delta'],
    ]

    f.writelines(LINES)
    f.close()
