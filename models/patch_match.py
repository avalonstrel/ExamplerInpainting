import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
def get_spatial_feat(feat, pos, psize):
    """
    Take the feat at on position
    Params:
        feat(torch.Tensor) :B*C*H*W
        pos(torch.Tensor): B*2
    Return: Batch feature(torch.Tensor): B*C*psize*psize
    """

    return torch.gather([feat[i, :, torch.max(0,pos[i, 0]-psize):torch.min(pos[i, 0]+psize,feat.size(2)),
        torch.max(pos[i, 1]-psize,0):torch.min(pos[i, 1]+psize, feat.size(3))] for i in range(feat.size(0))])

def get_patch_feats(feat1, feat2, feat1_, feat2_, pos1, pos2, psize=5):
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
    return torch.gather([f[i, pos[i, 0], pos[i, 1]]  for i in range(pos.size(0))])

def f_random_mapping(f, pos, r):
    """
    Implement f on position with a random search, return the position mapped by f
    """
    return torch.gather([f[i, pos[i, 0], pos[i, 1]]+(-1+np.random.random()*2)*r  for i in range(pos.size(0))])

def distance(feat1, feat2):
    """
    A distance function return shape B tensor
    """
    return F.pairwise_distance(feat1, feat2)

def feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos, psize):
    """
    Get the patch distance between two patch
    """
    pos_f = f_mapping(f, pos)
    pos_feat1, pos_feat2, pos_feat1_, pos_feat2_ = get_patch_feats(feat1, feat2, feat1_, feat2_, pos, pos_f, psize)
    best_dist = distance(pos_feat1, pos_feat2) + distance(pos_feat1_, pos_feat2_)
    return best_dist

def update_f_by_ori(pos, pos_new, dist, dist_new, f):
    """
    Given the distance function update the f and return the best_dist
    """
    dist_cmp = dist > dist_new
    best_dist = dist_new * dist_cmp + dist * (1 - dist_cmp)
    best_f = f.clone()
    for i in range(pos.size(0)):
        best_f[i][pos[i]] = f[i][(pos[i]*(1-dist_cmp[i])+pos_new[i]*dist_cmp[i])]
    #best_f[] = torch.gather([f[i][(pos[i]*(1-dist_cmp[i])+pos_new[i]*dist_cmp[i])] for i in range(pos.size(0))])
    return best_f, best_dist

def update_f_by_dest(pos, pos_d, dist, dist_new, f):
    """
    Given the distance function update the f and return the best_dist
    """
    dist_cmp = dist > dist_new
    best_dist = dist_new * dist_cmp + dist * (1 - dist_cmp)
    pos_f = f_mapping(f, pos)
    best_f = f.clone()
    for i in range(pos.size(0)):
        best_f[i][pos[i]] = pos_f[i]*(1-dist_cmp[i])+pos_d*dist_cmp[i]

    # best_f = torch.gather([pos_f[i]*(1-dist_cmp[i])+pos_d*dist_cmp[i] for i in range(pos.size(0))])
    return best_f, best_dist

def propagation(pos, change, f, feat1, feat2, feat1_, feat2_, psize=5):
    """
    Batch Propagation in patch match.
    Params:
        pos(torch.Tensor:B*2): batch of position
        change(torch.Tensor:2): direction for propagation
        f(torch.Tensor:B*H*W*2): a \phi_a->b function represented by a tensor relative position
        feat*(torch.Tensor:B*C*H*W): batch features
    """
    up_change, left_change  = torch.zeros_like(change), torch.zeros_like(change)
    up_change[0], left_change[1] = change
    # Batch pos adding up_change new B*2
    pos_new = pos + up_change
    best_dist = feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos, psize)
    new_dist = feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos_new, psize)
    best_f, best_dist = update_f_by_ori(pos, pos_new, best_dist, new_dist, f)
    # The same process but test left change
    pos_new = pos + left_change
    new_dist = feat_patch_distance(feat1, feat2, feat1_, feat2_, f, pos_new, psize)
    best_f, best_dist = update_f_by_ori(pos, pos_new, best_dist, new_dist, best_f)
    #dist_min, dist_ind = torch.min(dist, dist_new)
    return best_f, best_dist

def random_search(pos, f, best_f, best_dict, feat1, feat2, feat1_, feat2_, alpha=0.5, psize=5, omega=torch.tensor([256,256])):
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
        change = -add_change
        start = pos_end
        end = pos_start
    else:
        change = add_change
        start = pos_start
        end = pos_end
    return change, start, end

def deep_patch_match(feat1, feat2, feat1_, feat2_, psize=5, iteration=5, alpha=0.5):
    """
    A deep patch match method based on two pairs data. Formulated in Deep Image Analogy
    Original version only use img1 and img2
    Params: img1(torch.Tensor):  shape B*C*H*W
    """
    add_change = [1,1]
    pos_start = [0,0]
    pos_end = feat1.size()[2:] - 1
    f = torch.tensor(np.random.random((feat1.size(0), feat1.size(2), feat1.size(3), 2)))
    for i in range(iteration):
        change, start, end = initialize_direction(i, add_change, pos_start, pos_end)
        for x in range(int(start[0]), int(end[0]), step=int(change[0])):
            for y in range(int(start[1]), int(end[1]), step=int(change[1])):
                pos = torch.tensor([x,y])
                best_f, best_dict = propagation(pos, -change, f, feat1, feat2, feat1_, feat2_, psize)
                f, best_dict = random_search(pos, f, best_f, best_dict, feat1, feat2, feat1_, feat2_, psize=psize)
    return f

def reconstruct_avg(feat2, f, psize=5):
    """
    Reconstruct another batch feat1 from batch feat2 by f
    Params:
        feat2(torch.Tensor:shape (B*C*H*W)): feature 2
        f(torch.Tensor:shape (B*H*W*2)): f : 1->2
    """
    #assert feat.size()[2:] == f.size(H)
    feat1 = torch.zeros_like(feat2)

    for x in range(feat.size(2)):
        for y in range(feat.size(3)):
            pos = torch.zeros(feat2.size(0), 2).type(torch.LongTensor)
            pos = pos + torch.tensor([x,y]).type(torch.LongTensor)
            pos_f = f_mapping(f, pos)
            batch_feat = get_spatial_feat(feat2, pos_f, psize)
            b,c,hp,wp = batch_feat.size()
            feat1[:,:,x,y] = batch_feat.view(b,c,hp*wp).mean(dim=2)

    return feat1



def main():
    transforms_fun = transforms.Compose([transforms.ToTensor()])
    img1 = transforms_fun(Image.open('test1.png'))
    img2 = transforms_fun(Image.open('test2.png'))
    img1_ = transforms_fun(Image.open('test1_.png'))
    img2_ = transforms_fun(Image.open('test2_.png'))
    f = deep_patch_match(img1, img2, img1_, img2_, psize=5, iteration=5, alpha=0.5)
    img1 = reconstruct_avg(img2, f, psize=5)

if __name__ == '__main__':
    main()
