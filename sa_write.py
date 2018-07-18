import json
import operator


def print_dict(sa, filename='res.txt'):
    # read attr.
    with open('./sa/data/attr.txt', 'r') as rf:
        attr = rf.readlines()
        for i, val in enumerate(attr):
            if val[-1] == '\n':
                attr[i] = val[:-1]

    wf = open('./sa/'+filename, 'w')

    sa_tuple = sorted(sa.items(), key=operator.itemgetter(0))

    for idx, tup in enumerate(sa_tuple):
        tmp = []
        for i, _ in enumerate(tup[1]):
            if tup[1][i] == 1:
                tmp.append(attr[i])

        print('('+str(idx+1)+'/'+str(len(sa))+')', tup[0], ':', tmp)
        wf.write('(' + str(idx + 1) + '/' + str(len(sa)) + ') ' + tup[0] + ' : ' + (', '.join(tmp)).strip() + '\n')

    wf.close()


if __name__ == "__main__":
    names = ['close_0', 'open_0']

    for name in names:
        # read dict
        with open('./sa/'+name+'.json', 'r') as rf:
            sa = json.loads(rf.read())

        # save dict
        print_dict(sa, filename=name+'.txt')
