import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
SHAPE = (64,64)

def get_spatial_feat(feat, pos, psize):
    """
    Take the feat at on position
    Params:
        feat(torch.Tensor) :B*C*H*W
        pos(torch.Tensor): B*2
    Return: Batch feature(torch.Tensor): B*C*psize*psize
    """

    #print(pos)
    return torch.stack([feat[i, :, max(0,pos[i, 0]-psize):min(pos[i, 0]+psize+1,feat.size(2)),
        max(pos[i, 1]-psize,0):min(pos[i, 1]+psize+1, feat.size(3))] for i in range(feat.size(0))])

def get_patch_feats(feat1, feat2, feat1_, feat2_, pos1, pos2, psize=2):
    """
    Given the features and the position, return the 4 feat for compute energy function
    """
    pos_feat1 = get_spatial_feat(feat1, pos1, psize)
    pos_feat2 = get_spatial_feat(feat2, pos2, psize)
    pos_feat1_ = get_spatial_feat(feat1_, pos1, psize)
    pos_feat2_ = get_spatial_feat(feat2_, pos2, psize)
    return (pos_feat1, pos_feat2, pos_feat1_, pos_feat2_)

def f_mapping(f, pos):
    """
    Implement f on position, return the position mapped by f
    """
    #print("pos{}".format(pos.size(), f.size()))
    return torch.stack([f[i, pos[i, 0], pos[i, 1],:]  for i in range(pos.size(0))])

def f_random_mapping(f, pos, r):
    """
    Implement f on position with a random search, return the position mapped by f
    """
    #print(r)
    #print(((-1+torch.rand(2)*2)*r).type(torch.LongTensor))
    return torch.stack([(f[i, pos[i, 0], pos[i, 1],:]+((-1+torch.rand(2)*2)*r).type(torch.LongTensor)).clamp(2, SHAPE[0]-3)  for i in range(pos.size(0))])

def distance(feat1, feat2):
    """
    A distance function return shape B tensor
    """
    #print(feat1.size(), feat2.size())
    batch_size = feat1.size(0)
    min_h, min_w = min(feat1.size(2), feat2.size(2)), min(feat1.size(3), feat2.size(3))
    return F.pairwise_distance((feat1[:, :, :min_h, :min_w]).reshape(batch_size, -1), (feat2[:, :, :min_h, :min_w]).reshape(batch_size, -1))

def feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos, psize):
    """
    Get the patch distance between two patch
    """
    pos_f = f_mapping(f, pos)
    pos_feat1, pos_feat2, pos_feat1_, pos_feat2_ = get_patch_feats(feat1, feat2, feat1_, feat2_, pos, pos_f, psize)
    best_dist = distance(pos_feat1, pos_feat2) + distance(pos_feat1_, pos_feat2_)
    return best_dist

def update_f_by_ori(pos, pos_new, dist, dist_new, f, best_f):
    """
    Given the distance function update the f and return the best_dist
    """
    dist_cmp = (dist > dist_new).type(torch.FloatTensor)
    best_dist = dist_new * dist_cmp + dist * (1 - dist_cmp)
    #best_f = f.clone()
    dist_cmp = dist_cmp.type(torch.LongTensor)
    for i in range(pos.size(0)):
        best_f[i][pos[i]] = f[i][(pos[i]*(1-dist_cmp[i])+pos_new[i]*dist_cmp[i])]
    #best_f[] = torch.gather([f[i][(pos[i]*(1-dist_cmp[i])+pos_new[i]*dist_cmp[i])] for i in range(pos.size(0))])
    return best_f, best_dist

def update_f_by_dest(pos, pos_d, dist, dist_new, f):
    """
    Given the distance function update the f and return the best_dist
    """
    dist_cmp = (dist > dist_new).type(torch.FloatTensor)
    best_dist = dist_new * dist_cmp + dist * (1 - dist_cmp)
    pos_f = f_mapping(f, pos)
    best_f = f.clone()
    dist_cmp = dist_cmp.type(torch.LongTensor)
    for i in range(pos.size(0)):
        best_f[i][pos[i]] = pos_f[i]*(1-dist_cmp[i])+pos_d*dist_cmp[i]

    # best_f = torch.gather([pos_f[i]*(1-dist_cmp[i])+pos_d*dist_cmp[i] for i in range(pos.size(0))])
    return best_f, best_dist



def propagation(pos, change, f, dist_f, feat1, feat2, feat1_, feat2_, psize=2):
    """
    Batch Propagation in patch match.
    Params:
        pos(torch.Tensor:B*2): batch of position
        change(torch.Tensor:2): direction for propagation
        f(torch.Tensor:B*H*W*2): a \phi_a->b function represented by a tensor relative position
        dist_f(torch.Tensor:B*H*W): a \phi_a->b function represented by a tensor min dist
        feat*(torch.Tensor:B*C*H*W): batch features
    Return best_f(torch.Tensor:B*H*W*2) best_dist_f(torch.Tensor:B*H*W)
    """
    up_change, left_change  = torch.zeros_like(torch.tensor(change)), torch.zeros_like(torch.tensor(change))
    up_change[0], left_change[1] = change
    # Batch pos adding up_change new B*2
    pos_new = pos + up_change
    #ori_dist = feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos, psize)
    new_dist = feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos_new, psize)
    dist_f = update_dist_f(new_dist, dist_f, pos, pos_new)
    best_f = f.clone()
    best_f, best_dist = update_f_by_ori(pos, pos_new, dist_f, new_dist, f, best_f)
    # The same process but test left change
    pos_new = pos + left_change
    new_dist = feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos_new, psize)
    best_f, best_dist = update_f_by_ori(pos, pos_new, best_dist, new_dist, f, best_f)
    #dist_min, dist_ind = torch.min(dist, dist_new)
    return best_f, best_dist

def random_search(pos, f, best_f, best_dist, feat1, feat2, feat1_, feat2_, alpha=0.5, psize=2, omega=torch.tensor([32.,32.])):
    """
    Batch Random Search For patch Match
    """
    r = omega

    while r[0] > 1 and r[1] > 1:
        pos_random_f = f_random_mapping(f, pos, r)
        pos_rand_feat1, pos_rand_feat2, pos_rand_feat1_, pos_rand_feat2_ = get_patch_feats(feat1, feat2, feat1_, feat2_, pos, pos_random_f, psize)
        dist_rand = distance(pos_rand_feat1, pos_rand_feat2) + distance(pos_rand_feat1_, pos_rand_feat2_)
        best_f, best_dist = update_f_by_dest(pos, pos_random_f, best_dist, dist_rand, best_f)
        r = alpha*r

    return best_f, best_dist

def initialize_direction(i, add_change, pos_start, pos_end):
    if (i+1) % 2 == 0:
        change = [-add_change[0],-add_change[1]]
        start = pos_end
        end = pos_start
    else:
        change = add_change
        start = pos_start
        end = pos_end
    return change, start, end

def deep_patch_match(feat1, feat2, feat1_, feat2_, psize=2, iteration=5, alpha=0.5):
    """
    A deep patch match method based on two pairs data. Formulated in Deep Image Analogy
    Original version only use img1 and img2
    Params: img1(torch.Tensor):  shape B*C*H*W
    """
    add_change = [1,1]
    pos_start = [0,0]
    pos_end = [SHAPE[0]-1, SHAPE[1]-1] #list(torch.tensor(feat1.size()[2:]).type(torch.LongTensor) - 1)
    print(feat1.size())
    f = torch.tensor(np.random.randint(0, SHAPE[0]-1, size=(feat1.size(0), feat1.size(2), feat1.size(3), 2)))
    for i in range(iteration):
        print("Iteration {}: Running".format(i+1))
        change, start, end = initialize_direction(i, add_change, pos_start, pos_end)
        print('start:{}, end:{}, change:{}'.format(start, end, change))
        end_time = time.time()
        for x in range(int(start[0]+change[0]*psize), int(end[0]-change[0]*psize), int(change[0])):
            for y in range(int(start[1]+change[0]*psize), int(end[1]-change[0]*psize), int(change[1])):
                pos = torch.tensor([x,y]).view(1,2).repeat(feat1.size(0), 1)

                best_f, best_dist = propagation(pos, [change[0], change[1]], f, best_dist, feat1, feat2, feat1_, feat2_, psize)

                #end_time = time.time()
                f, best_dist = random_search(pos, f, best_f, best_dist, feat1, feat2, feat1_, feat2_, psize=psize)
                #print("Random Search Time :{}".format(time.time()-end_time))
        print("Iteration {}: Finishing Time : {}".format(i+1, time.time()-end_time))
    return f

def reconstruct_avg(feat2, f, psize=2):
    """
    Reconstruct another batch feat1 from batch feat2 by f
    Params:
        feat2(torch.Tensor:shape (B*C*H*W)): feature 2
        f(torch.Tensor:shape (B*H*W*2)): f : 1->2
    """
    #assert feat.size()[2:] == f.size(H)
    feat1 = torch.zeros_like(feat2)

    for x in range(feat2.size(2)):
        for y in range(feat2.size(3)):
            pos = torch.zeros(feat2.size(0), 2).type(torch.LongTensor)
            pos = pos + torch.tensor([x,y]).type(torch.LongTensor)
            pos_f = f_mapping(f, pos)
            print(pos_f)
            batch_feat = get_spatial_feat(feat2, pos_f, psize)
            b,c,hp,wp = batch_feat.size()
            feat1[:,:,x,y] = batch_feat.view(b,c,hp*wp).mean(dim=2)

    return feat1

def reshape_test(img):
    return img.view(1, *img.size())

def main():
    transforms_fun = transforms.Compose([transforms.Resize(SHAPE),transforms.ToTensor()])
    img1 = transforms_fun(Image.open('../test1.png'))
    img2 = transforms_fun(Image.open('../test2.png'))
    img1_ = transforms_fun(Image.open('../test1_.png'))
    img2_ = transforms_fun(Image.open('../test2_.png'))
    img1, img2, img1_, img2_ = reshape_test(img1),reshape_test(img2),reshape_test(img1_),reshape_test(img2_)
    f = deep_patch_match(img1, img2, img1_, img2_, psize=2, iteration=5, alpha=0.5)
    img1 = reconstruct_avg(img2, f, psize=2)
    img1 = img1.transpose(1,2).transpose(2,3)*255
    print(img1.size())
    print(f, img1)
    img1_ = Image.fromarray(img1[0].numpy().astype(np.uint8))
    img1_.save("out_test1.png")

if __name__ == '__main__':
    main()
