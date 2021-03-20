

f = open("./data/CelebA/list_attr_celeba.txt")
glass_txt = "./data/CelebA/glass.txt"
newf = open(glass_txt, "w+")
noGlass_txt = "./data/CelebA/noGlass.txt"
newNof = open(noGlass_txt, "w+")

line = f.readline()
line = f.readline()
line = f.readline()
while line:
    array = line.split()
    if array[0] == "000154.jpg":
        print(array[16])

    if array[16] == "-1":
        new_context = array[0] + '\n'
        newNof.write(new_context)
    else:
        new_context = array[0] + '\n'
        newf.write(new_context)
    line = f.readline()

newf.seek(0)  # 指针指向文件头
lines = len(newf.readlines())
print("There are %d lines in %s" % (lines, glass_txt))
newNof.seek(0)
lines = len(newNof.readlines())
print("There are %d lines in %s" % (lines, noGlass_txt))


f.close()
newf.close()
newNof.close()