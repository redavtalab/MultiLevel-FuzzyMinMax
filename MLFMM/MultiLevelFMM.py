# import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import torch

class MLFMM:
    def __init__(self, theta=0.5, gamma=1, mu=.25, no_levels=3, random_state=0):
        self.theta = theta
        self.gamma = gamma
        self.mu = mu
        self.no_levels = no_levels
        self.random_state = random_state
        self.hbsV = None
        self.hbsW = None
        self.Cls = None
        self.S_netID = None

        self.olsV = None
        self.olsW = None

        self.levels = None
        self.no_features = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def is_expandable(self, boxInd, x):
        candV = torch.minimum(self.hbsV[boxInd], x)
        candW = torch.maximum(self.hbsW[boxInd], x)
        return all((candW-candV) < self.theta)

    def expand_box(self, boxInd, x):
        self.hbsV[boxInd] = torch.minimum(self.hbsV[boxInd],x)
        self.hbsW[boxInd] = torch.maximum(self.hbsW[boxInd], x)

    def membership(self,x_q, subV,subW):
        no_class_boxes , no_features = torch.tensor(subV.size(), device=self.device)

        hboxC = torch.add(subV, subW) / 2
        boxes_W = subW.reshape([no_class_boxes, 1, no_features])
        boxes_V = subV.reshape(no_class_boxes, 1, no_features)
        boxes_center = hboxC.reshape(no_class_boxes, 1, no_features)
        hDistance = ((boxes_W - boxes_V) / 2).reshape(no_class_boxes, 1, no_features)
        d = torch.abs(boxes_center - x_q) - hDistance
        d[d < 0] = 0
        dd = torch.linalg.norm(d, axis=2)
        dd = dd / torch.sqrt(no_features)
        m = 1 - dd  # m: membership
        m = torch.pow(m, 6)
        return m

    def overlaps(self, vj, wj, vk, wk):
        '''
        Check if any classwise dissimilar hyperboxes overlap
        '''


        delta_new = delta_old = 1
        min_overlap_index = -1
        for i in range(len(vj)):
            if vj[i] < vk[i] < wj[i] < wk[i]:
                delta_new = min(delta_old, wj[i] - vk[i])

            elif vk[i] < vj[i] < wk[i] < wj[i]:
                delta_new = min(delta_old, wk[i] - vj[i])

            elif vj[i] < vk[i] < wk[i] < wj[i]:
                delta_new = min(delta_old, min(wj[i] - vk[i], wk[i] - vj[i]))

            elif vk[i] < vj[i] < wj[i] < wk[i]:
                delta_new = min(delta_old, min(wj[i] - vk[i], wk[i] - vj[i]))

            else:  #There is no overlaps between two boxes
                min_overlap_index = -1
                break

            if delta_old - delta_new > 0:
                min_overlap_index = i
                delta_old = delta_new

        if min_overlap_index >= 0:
               return False
        return True


    def setup_hyperboxs(self, X,y):
        no_samples, no_features = X.shape

        boxesV = torch.empty(0, no_features, device=self.device)
        boxesW = torch.empty(0, no_features, device=self.device)
        boxesCls = torch.empty(0, device=self.device)

        for ind, x_t in enumerate(X):
            curClass = y[ind]
            if len(boxesCls)==0 or torch.sum(boxesCls==curClass)==0: # the first sample has been used to create the first box.
                # Create the first box:
                box = x_t.reshape(1,no_features)
                boxesV = torch.cat((boxesV, box))
                boxesW = torch.cat((boxesW, box))
                boxesCls = torch.cat((boxesCls, curClass.reshape(1)))
                continue
            # Filter the same class's hyperboxes
            class_boxes_ind = boxesCls == curClass
            class_boxes_V = boxesV[class_boxes_ind]
            class_boxes_W = boxesW[class_boxes_ind]

            # Is x_t located inside a box?
            memberships = self.membership(x_t, class_boxes_V,class_boxes_W)
            if torch.max(memberships) >= 1:
                continue

            ######################## Expand ############################

            # Sort boxes by the distances
            hboxC = (class_boxes_V + class_boxes_W) / 2
            expanded = False
            dist_to_boxes = torch.linalg.norm(x_t - hboxC, axis=1)

            indexes = torch.argsort(dist_to_boxes)

            for ind in indexes:

                candV = torch.minimum(class_boxes_V[ind], x_t)
                candW = torch.maximum(class_boxes_W[ind], x_t)
                if all((candW - candV) < self.theta):
                    class_boxes_V[ind] = candV
                    class_boxes_W[ind] = candW
                    boxesV[class_boxes_ind] = class_boxes_V
                    boxesW[class_boxes_ind] = class_boxes_W
                    expanded = True
                    break


            ######################## Creation ############################
            #  If any hyperbox didn't expand
            if expanded == False:
                box = x_t.reshape(1, no_features)
                boxesV = torch.cat((boxesV, box))
                boxesW = torch.cat((boxesW, box))
                boxesCls = torch.cat((boxesCls, curClass.reshape(1)))

        return boxesV, boxesW, boxesCls

    def fit(self, X, y):
        # implementation of the fit method
        # should train the model on the data (X, y) and store any necessary information

        X=torch.tensor(X,device=self.device)
        y=torch.tensor(y,device=self.device)

        if X.min() < 0 or X.max() > 1:
            print("** Normalizing data prior to training is recommended **")

        no_samples, no_features = X.shape

        # HBS --> Hyperboxes
        self.hbsV = torch.empty(0, no_features, device=self.device)
        self.hbsW = torch.empty(0, no_features, device=self.device)
        self.Cls = torch.empty(0, device=self.device)
        self.S_netID = torch.empty(0, device=self.device)
        # self.S_netID = torch.tensor([0], device=self.device)

        # OLS --> overlap boxes
        self.olsV = torch.empty(0, no_features, device=self.device)
        self.olsW = torch.empty(0, no_features, device=self.device)
        self.levels = torch.tensor([0], device=self.device)
        self.olsV = torch.concat((self.olsV, torch.min(X,0).values.reshape(1,no_features)))
        self.olsW = torch.concat((self.olsW, torch.max(X,0).values.reshape(1,no_features)))
        cur_snet = 0
        level = 0
        while(level < self.no_levels):
            # Determine the hyperboxes of previous level
            level_ind = self.levels==level

            cur_snet_V = self.olsV[level_ind]
            cur_snet_W = self.olsW[level_ind]

            for olBox_v, olBox_w in zip(cur_snet_V,cur_snet_W):

                inds = torch.logical_and(torch.all(olBox_v <= X, dim=1), torch.all(X <= olBox_w, dim=1))
                sub_X = X[inds]
                sub_y = y[inds]
                # Generating hyperboxes of one node in the next level
                boxesV, boxesW, boxesCls = self.setup_hyperboxs(sub_X, sub_y)

                self.hbsV = torch.concat((self.hbsV, boxesV))
                self.hbsW = torch.concat((self.hbsW, boxesW))
                self.Cls = torch.concat((self.Cls, boxesCls))
                # new_ol_id = len(self.olsV) - 1  # torch.max(self.S_netID)
                self.S_netID = torch.concat((self.S_netID, torch.ones((boxesV.size(0)), device=self.device) * cur_snet))
                cur_snet += 1
                for ind, (bv, bw, cls) in enumerate(zip(boxesV, boxesW, boxesCls)):
                    # Find all overlaps (include the same class)
                    maxofmins = torch.max(bv, boxesV[ind:])  # a box vs an array
                    minofmaxs = torch.min(bw, boxesW[ind:])  # a box vs an array
                    overlaps = torch.all(minofmaxs >= maxofmins, dim=1)
                    # Filter the same class's boxes
                    overlaps_inds = torch.logical_and(overlaps, boxesCls[ind:] != cls)
                    if torch.sum(overlaps_inds)==0:   # if there is no overlap
                        continue
                    # Determine the min and max points of overlaps
                    overlaps_V = maxofmins[overlaps_inds]
                    overlaps_W = minofmaxs[overlaps_inds]

                    self.olsV = torch.concat((self.olsV, overlaps_V))
                    self.olsW = torch.concat((self.olsW, overlaps_W))
                    self.levels = torch.concat((self.levels,torch.ones((overlaps_V.size(0)), device=self.device) * level+1))

            self.theta = self.theta * self.mu
            level +=1


    def predict(self, X):
        # implementation of the predict method
        # should return the predicted labels for the input data X
        X = torch.cuda.FloatTensor(X)

        no_sample_test, no_features = X.shape

        # Are all samples inside the feature space?
        self.olsV[0] = torch.min(self.olsV[0], torch.min(X, 0).values)
        self.olsW[0] = torch.max(self.olsV[0], torch.max(X, 0).values)

        X=torch.cuda.FloatTensor(X)
        predicted_labels = torch.zeros((X.size(0),1),device=self.device)
        for ind, x_q in enumerate(X):
            insideInds = torch.logical_and(torch.all(self.olsV <= x_q, dim=1), torch.all(x_q <= self.olsW, dim=1))

            iD = torch.max(torch.where(insideInds)[0])
            hbox_inds = self.S_netID == iD
            ## if there is no boxesV related to iD, the next s_net should be selected (second from the end)
            while torch.sum(hbox_inds) == 0:
                insideInds[iD] = False
                iD = torch.max(torch.where(insideInds)[0])
                hbox_inds = self.S_netID == iD

            boxesV = self.hbsV[hbox_inds]
            boxesW = self.hbsW[hbox_inds]
            boxesCls = self.Cls[hbox_inds]
            memberships = self.membership(x_q, boxesV, boxesW)
            max_ind = torch.argmax(memberships)
            predicted_labels[ind] = boxesCls[max_ind]


        return predicted_labels

    def score(self, X, y):
        X = torch.cuda.FloatTensor(X)
        y = torch.cuda.FloatTensor(y)

        # implementation of the score method
        # should return the mean accuracy on the given test data and labels
        y_pred = self.predict(X)
        return torch.sum(y_pred.reshape(len(y),) == y) / len(y)




