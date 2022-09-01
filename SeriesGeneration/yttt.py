filename_dataset = 'file.txt'
ap =[]
app = []
dataset = open(filename_dataset, 'r', encoding='utf-8')
dataset = dataset.read()
mrLines = dataset.split('\n')
for a in mrLines:
    a.split(' ')
    ap = a.split(' ')
    app = ap[1:]
    print(app)
    dataset1 = open('file1.txt', 'a', encoding='utf-8')
    for i in app:
        dataset1.write(i + ' ')
    dataset1.write('\n')
    # print(app)
print(mrLines)