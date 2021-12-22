import numpy as np
import pandas as pd
from PIL import Image
import cv2
from os.path import join
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

class DataGraph:
    def __init__(self, pairs_file):
        self.pairs_file = pairs_file
        self.data = pd.DataFrame()
        self.nqueries = 0
        self.ntotal = 0

    def update_graph(self):
        with open(self.pairs_file, "r") as f:
            pairs = []
            for line in f:
                line = line.strip()
                pairs.append(line.split(" "))
            self.ntotal = len(pairs)
            self.data = pd.DataFrame(data=pairs,columns=["query","db"], dtype=str).groupby('query')['db'].apply(list).reset_index(name="matches")
            self.nqueries = self.data["query"].size
            # print(self.data)
            # for i in self.data["matches"]:
            #     print(len(i))
            
    def get_data(self, idx):
        n_matches_per_query = self.ntotal//self.nqueries
        id_q = idx//n_matches_per_query
        id_m = idx%n_matches_per_query
        if id_m == 0:
            return self.data["query"][id_q], id_q
        else:
            return self.data["matches"][id_q][id_m], id_q
    
    def get_data_rand(self, idx):
        # Note: Assumes equal number of matches for each query e.g. 50 each for 922 Aachen query images 
        n_matches_per_query = self.ntotal//self.nqueries
        id_q = idx//n_matches_per_query
        id_m = idx%n_matches_per_query
        id_rq = np.random.randint(self.nqueries)
        while (id_q==id_rq):
            id_rq = np.random.randint(self.nqueries)
        id_rm = np.random.randint(n_matches_per_query)
        # print(self.nqueries, id_rq)
        # print(self.data["query"][id_q])
        # print(self.data["matches"][id_q][id_m])
        # print(self.data["matches"][id_rq][id_rm])
        return  self.data["query"][id_q], self.data["matches"][id_q][id_m], self.data["matches"][id_rq][id_rm]

    def get_len(self):
        return self.ntotal
    
    def get_class_size(self):
        return self.nqueries

class MetricLearningTriplets(Dataset):
    def __init__(self, data_root, pairs_path):
        self.root_dir = data_root
        self.pairs_path = pairs_path
        self.dg = DataGraph(self.pairs_path)
        self.dg.update_graph()

    def __len__(self):
        return self.dg.get_len()

    def __getitem__(self, idx):
        impth_q, impth_pos, impth_neg = self.dg.get_data_rand(idx)
        im_q = Image.open(join(self.root_dir, impth_q)).convert(mode="L")
        im_pos = Image.open(join(self.root_dir, impth_pos)).convert(mode="L")
        im_neg = Image.open(join(self.root_dir, impth_neg)).convert(mode="L")
        
        orb = cv2.ORB_create(nfeatures=100)
        im_q = np.array(im_q)
        kp = orb.detect(im_q, None)
        kp, des_q = orb.compute(im_q, kp)
        # im_q = cv2.drawKeypoints(im_q, kp, None, color=(0,255,0), flags=0)
        
        im_pos = np.array(im_pos)
        kp = orb.detect(im_pos, None)
        kp, des_pos = orb.compute(im_pos, kp)
        # im_pos = cv2.drawKeypoints(im_pos, kp, None, color=(0,255,0), flags=0)

        im_neg = np.array(im_neg)
        kp = orb.detect(im_neg, None)
        kp, des_neg = orb.compute(im_neg, kp)
        # im_neg = cv2.drawKeypoints(im_neg, kp, None, color=(0,255,0), flags=0)
        
        # print(des_q.shape)
        # print(des_pos.shape)
        # print(des_neg.shape)
        
        # _, axes = plt.subplots(1,3)
        # axes[0].imshow(im_q, cmap="gray")
        # axes[0].set_title("Query")
        # axes[1].imshow(im_pos, cmap="gray")
        # axes[1].set_title("Positive")
        # axes[2].imshow(im_neg, cmap="gray")
        # axes[2].set_title("Negative")
        # plt.show()

        return Tensor(des_q), Tensor(des_pos), Tensor(des_neg)
    
class MetricLearningClasses(Dataset):
    def __init__(self, data_root, pairs_path):
        self.root_dir = data_root
        self.dg = DataGraph(pairs_path)
        self.dg.update_graph()
        
    def __len__(self):
        return self.dg.get_len()
    
    def __getitem__(self,idx):
        impth, lbl = self.dg.get_data(idx)
        im = Image.open(join(self.root_dir, impth)).convert(mode="L")
        orb = cv2.ORB_create(nfeatures=100)
        im = np.array(im)
        kp = orb.detect(im, None)
        kp, des = orb.compute(im, kp)
        return Tensor(des), lbl
        
    
def load_data(dataset_root, pairs_path, num_workers=0, batch_size=128, triplets=False):
    if triplets:
        ds = MetricLearningTriplets(dataset_root, pairs_path)
    else:
        ds = MetricLearningClasses(dataset_root, pairs_path)
    return DataLoader(ds, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    
if __name__ == "__main__":
    # ds = MetricLearningDataset("datasets/Aachen-Day-Night/images/images_upright",
    #                             "pairs/aachen/pairs-query-netvlad50.txt") 
    # for d1,d2,d3 in ds:
    #     print(d1.shape)
    #     print(d2.shape)
    #     print(d3.shape)
    #     break
    ds = MetricLearningClasses("datasets/Aachen-Day-Night/images/images_upright",
                                "pairs/aachen/pairs-query-netvlad50.txt") 
    for des,lbl in ds:
        print(des.shape)
        print(lbl.shape)
        break