import os


datapath = 'D:\\Databases\\BosphorusSign\\General_frames'
classdir = sorted(list(map(int, os.listdir(datapath))))

class_indices = {item: i for i, item in enumerate(classdir, 0)}
class_indices_file = open('../list/general/class_indices.list', 'w')
for item, idx in class_indices.items():
    class_indices_file.write(str(item) + ':' + str(idx) + '\n')

train_split_file = open('../list/general/train.list', 'w')
test_split_file = open('../list/general/test.list', 'w')

for _clazz in classdir:
    _clazzpath = os.path.join(datapath, str(_clazz))

    userdir = sorted(os.listdir(_clazzpath))
    for _user in userdir:
        _user_split = _user.split('_')

        print(str(_clazz) + '-' + _user)
        if _user_split[1] != '4':
            train_split_file.write(os.path.join(datapath, str(_clazz), _user) + ' ' + str(class_indices[_clazz]) + '\n')
        else:
            test_split_file.write(os.path.join(datapath, str(_clazz), _user)  + ' ' + str(class_indices[_clazz]) + '\n')

train_split_file.close()
test_split_file.close()
class_indices_file.close()
