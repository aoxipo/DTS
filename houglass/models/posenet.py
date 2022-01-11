import torch
from torch import nn
from .layers import Conv, Hourglass, Pool, Residual, DWConv, conv1x1

from .task.loss import HeatmapLoss

use_gpu = torch.cuda.is_available()

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()
    

    def forward(self, imgs):
        ## our posenet
        x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss


class MYDensityNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, Conv_method = "Conv",image_shape = (1,256,256),bn=False, increase=0, **kwargs):
        super(MYDensityNet, self).__init__()
        self.Conv_method = Conv_method
        self.image_shape = image_shape
        self.nstack = nstack
        self.inp_dim = inp_dim 
        self.oup_dim = oup_dim
        
        self.conv_type_dict = {
            "DWConv":DWConv,
            "Conv":Conv,
        }
            
        print("using :",Conv_method)
        self.pre = nn.Sequential(
            self.conv_type_dict[self.Conv_method](image_shape[0], 32, 7, 2, bn=True, relu=True).cuda(),
            Residual(32, 64, self.conv_type_dict[self.Conv_method]),
            Pool(2, 2),
            #Residual(64, 128, self.conv_type_dict[self.Conv_method]),
            Residual(64, inp_dim, self.conv_type_dict[self.Conv_method])
        ).cuda()
       
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] ).cuda()
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim, self.conv_type_dict[self.Conv_method]),
            self.conv_type_dict[self.Conv_method](inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] ).cuda()
        
        # self.up = conv1x1(1,1)
        
        self.outs = nn.ModuleList( [self.conv_type_dict[self.Conv_method](inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] ).cuda()
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] ).cuda()
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] ).cuda()
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss().cuda()

    def forward(self, imgs):
        ## our posenet
        P,C,W,H = imgs.size()

        if( C == 1 or C == 3):
            x = imgs
        else:
            x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        
        x_backup = x
        #print(x.size())
        x = self.pre(x)
        # print(x.size())
        combined_hm_preds = []
        attention_list = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)

            # print("hg:",hg.size()[-1],hg.size()[-2])
            
            attention = self.conv_type_dict[self.Conv_method](self.inp_dim, 1, kernel_size = 1, bn=True, relu=True)(hg)
            attention_list.append(attention)
            # print("attention:",attention.size())
            # attention = torch.flatten(hg,1)
            # attention = nn.Linear(in_features = attention.size()[1], out_features =hg.size()[-1] * hg.size()[-2] )(attention) 
            # # print("attention:",attention.size())
            # attention_list.append( nn.Linear(in_features = attention.size()[1], out_features = 1)(attention) )
            
            feature = self.features[i](hg)
            # print("feature:",feature.size())
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            # print("preds:", preds.size())
            if i < self.nstack - 1:
                # print(x.size(), self.merge_preds[i](preds).size(), self.merge_features[i](feature).size())
                # assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        Patch, C, H, W = attention.size()
        
        w = self.image_shape[1]/W
        self.up =  nn.Upsample(scale_factor= w , mode='nearest')
        multi_map = torch.zeros((Patch,C,int(w*H),int(w*W))).cuda()
 
        for i in range(self.nstack):
            #print(combined_hm_preds[i].size(), attention_list[i].size())
            up_preds = self.up(combined_hm_preds[i])
            up_attention = self.up(attention_list[i])
            # print(up_attention.size(),up_preds.size())
            #print( multi_map.size() )
            multi_map += torch.mul(up_preds, up_attention)
       
        multi_map = torch.mul(x_backup, multi_map)

        return multi_map #, torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss
