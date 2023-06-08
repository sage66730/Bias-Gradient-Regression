from genericpath import isdir
import os
import random
import pickle
import shutil
import argparse

def random_idx(all, train, valid):
    pool = [i for i in range(all)]
    train_set = random.sample(pool, train)
    pool = list(set(pool)-set(train_set))
    valid_set = random.sample(pool, valid)
    test_set = list(set(pool)-set(valid_set))

    return train_set, valid_set, test_set

def trajectory_idx(all, train, valid):
    train_set = [i for i in range(train)]
    valid_set = [i for i in range(train, train+valid)]
    test_set =  [i for i in range(train+valid, all)]

    return train_set, valid_set, test_set

def random_percent_idx(all, train, valid):
    pool = [i for i in range(all)]
    train_set = random.sample(pool, train)
    pool = list(set(pool)-set(train_set))
    valid_set = random.sample(pool, valid)
    test_set = list(set(pool)-set(valid_set))

    return train_set, valid_set, test_set

def trajectory_percent_idx(all, train, valid):
    assert train+valid < 1000
    train_num = int(all*(train/100))
    valid_num = int(all*(valid/100))

    train_set = [i for i in range(train_num)]
    valid_set = [i for i in range(train_num, train_num+valid_num)]
    test_set =  [i for i in range(train_num+valid_num, all)]

    return train_set, valid_set, test_set

def main(args):
    sample_num = {"benzene":627983, "uracil":133770, "naphthalene":326250, "aspirin":211762, "salicylic":320231, "malonaldehyde":993237, "ethanol":555092, "toluene":442790}

    if not os.path.isdir(f"{args.style}"): os.mkdir(f"{args.style}")
    if os.path.isdir(f"{args.style}/{args.train}"): 
        print("over writing split")
        shutil.rmtree(f"./{args.style}/{args.train}")
    os.mkdir(f"{args.style}/{args.train}")

    for m in sample_num.keys():
        os.mkdir(f"{args.style}/{args.train}/{m}")
        assert args.train+args.valid < sample_num[m]

        if args.train >= 100:
            if args.style == "random": train_set, valid_set, test_set = random_idx(sample_num[m], args.train, args.valid)
            elif args.style == "trajectory": train_set, valid_set, test_set = trajectory_idx(sample_num[m], args.train, args.valid)
            else: train_set, valid_set, test_set = [], [], []
        else:
            if args.style == "random": train_set, valid_set, test_set = random_percent_idx(sample_num[m], args.train, args.valid)
            elif args.style == "trajectory": train_set, valid_set, test_set = trajectory_percent_idx(sample_num[m], args.train, args.valid)
            else: train_set, valid_set, test_set = [], [], []

        assert len(train_set) and len(valid_set) and len(test_set)
        assert len(train_set)+len(valid_set)+len(test_set) == sample_num[m]

        with open(f"{args.style}/{args.train}/{m}/train.pkl", "wb") as f:
            pickle.dump(train_set, f)
        with open(f"{args.style}/{args.train}/{m}/valid.pkl", "wb") as f:
            pickle.dump(valid_set, f)
        with open(f"{args.style}/{args.train}/{m}/test.pkl", "wb") as f:
            pickle.dump(test_set, f)

    print("split generated")
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="trajectory", help="specify how to split the data")
    parser.add_argument("--train", type=int, default=1000, help="number of sample in training set")
    parser.add_argument("--valid", type=int, default=100, help="number of sample in validation set")


    main(parser.parse_args())