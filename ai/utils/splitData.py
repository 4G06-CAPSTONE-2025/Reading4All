import splitfolders


input_fold_path = ""
splitfolders.ratio(input_fold_path, output="",
    seed=133, ratio=(.8, .1, .1), group_prefix=None, move=False)