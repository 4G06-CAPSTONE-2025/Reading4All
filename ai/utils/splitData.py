import splitfolders

'''
input_folder/
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        imgA.jpg
        imgB.jpg
        ...

output/
    train/
        class1/
        class2/
    val/
        class1/
        class2/
    test/
        class1/
        class2/

seed can be changed and path for input and output folder needs to be given. Input folder shoud be same structure as above
'''

input_fold_path = ""
splitfolders.ratio(input_fold_path, output="",
    seed=133, ratio=(.8, .1, .1), group_prefix=None, move=False)