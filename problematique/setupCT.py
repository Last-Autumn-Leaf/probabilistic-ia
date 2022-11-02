from helpers.custom_class import ClassesTracker
import pickle

from helpers.custom_helper import DEFAULT_CT_PATH

if __name__=='__main__':
    CT=ClassesTracker()

    with open(DEFAULT_CT_PATH, 'wb') as inp:
        pickle.dump(CT,inp)
        print('dumped at',DEFAULT_CT_PATH)

