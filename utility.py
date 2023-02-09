import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from TorchDiffEqPack import odesolve
import sys
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from torchdiffeq import odeint
from functools import partial
import argparse
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = "Name of the data set. Options: EMT; Lineage; Bifurcation; Simulation", default = 'EMT')
    parser.add_argument('--timepoints', help = "time points of data", type=list, default = [0, 0.1, 0.3, 0.9, 2.1])
    parser.add_argument('--niters',help = "Number of traning iterations", type=int, default=5000)
    parser.add_argument('--lr', help = "Learning rate", type=float, default=3e-3) 
    parser.add_argument('--num_samples', help = "Number of sampling points i.e. batch size", type=int, default=100)
    parser.add_argument('--hidden_dim', help = "dimension of hidden layer", type=int, default=16)
    parser.add_argument('--n_hiddens', help = "number of hidden layers", type=int, default=4)
    parser.add_argument('--activation', type=str, default='Tanh')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_path', type=str,help="Input Files Directory. Default 'Input/'",default='Input/')
    parser.add_argument('--results_dir', type=str,help="Output Files Directory", default="Results/")
    parser.add_argument('--seed', help="random seed",type=int, default= 15)
    args = parser.parse_args()
    return args



class UOT(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.hyper_net1 = HyperNetwork1(in_out_dim, hidden_dim, n_hiddens,activation) #v= dx/dt
        self.hyper_net2 = HyperNetwork2(in_out_dim, hidden_dim, activation) #g

    def forward(self, t, states):
        z = states[0]
        g_z = states[1]
        logp_z = states[2]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            dz_dt = self.hyper_net1(t, z)
            
            g = self.hyper_net2(t, z)

            dlogp_z_dt = g - trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, g, dlogp_z_dt)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork1(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x
    

class HyperNetwork2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
        
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def MultimodalGaussian_density(x,time_all,time_pt,data_train,sigma,device):
    """density function for MultimodalGaussian
    """
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    for i in range(num_gaussian):
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i,:], sigma_matrix)
        p_unn = p_unn + torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    p_n = p_unn/num_gaussian
    return p_n


    
def Sampling(num_samples,time_all,time_pt,data_train,sigma,device):
    #perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    # check if number of points is <num_samples
    
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples)] + noise_add
    else:
        samples = mu[random.sample(range(0,num_gaussian), num_samples)] + noise_add
    return samples


def loaddata(args,device):
    data=np.load(os.path.join(args.input_path,(args.dataset+'.npy')),allow_pickle=True)
    data_train=[]
    for i in range(data.shape[1]):
        data_train.append(torch.from_numpy(data[0,i]).type(torch.float32).to(device))
    return data_train


def ggrowth(t,y,func,device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)
    y_00 = torch.zeros(y[1].shape).type(torch.float32).to(device)                       
    gg = func.forward(t, y)[1]
    return (y_0,y_00,gg)
    
    
def trans_loss(t,y,func,device):
    v = func.forward(t, y)[0]
    g = func.forward(t, y)[1]
    y_0 = torch.zeros(g.shape).type(torch.float32).to(device)
    y_00 = torch.zeros(v.shape).type(torch.float32).to(device)
    g_growth = partial(ggrowth,func=func,device=device)
    if torch.is_nonzero(t):
        _,_, exp_g = odeint(g_growth, (y_00,y_0,y_0), torch.tensor([0,t]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': 0.1})
        f_int = (torch.norm(v,dim=1)**2+torch.norm(g,dim=1)**2).unsqueeze(1)*torch.exp(exp_g[-1])
        return (y_00,y_0,f_int)
    else:
        return (y_00,y_0,y_0)


def train_model(mse,func,itr,args,data_train,train_time,integral_time,sigma_now,options,device):

    loss = 0
    L2_value1 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    L2_value2 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    
    for i in range(len(train_time)-1): 
        x = Sampling(args.num_samples, train_time,i+1,data_train,0.02,device)
        x.requires_grad=True
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)
        g_t1 = logp_diff_t1
        options.update({'t0': integral_time[i+1]})
        options.update({'t1': integral_time[0]})
        z_t0, g_t0, logp_diff_t0 = odesolve(func,y0=(x, g_t1, logp_diff_t1),options=options)
        aa = MultimodalGaussian_density(z_t0, train_time, 0, data_train,sigma_now,device) #normalized density
        
        zero_den = (aa < 1e-16).nonzero(as_tuple=True)[0]
        aa[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        logp_x = torch.log(aa)-logp_diff_t0.view(-1)
        
        aaa = MultimodalGaussian_density(x, train_time, i+1, data_train,sigma_now,device) * torch.tensor(data_train[i+1].shape[0]/data_train[0].shape[0]) # mass
        
        L2_value1[0][i] = mse(aaa,torch.exp(logp_x.view(-1)))
        
        loss = loss  + L2_value1[0][i]*1e4 
        
        # loss between each two time points
        options.update({'t0': integral_time[i+1]})
        options.update({'t1': integral_time[i]})
        z_t0, g_t0, logp_diff_t0= odesolve(func,y0=(x, g_t1, logp_diff_t1),options=options)
        
        aa = MultimodalGaussian_density(z_t0, train_time, i, data_train,sigma_now,device)* torch.tensor(data_train[i].shape[0]/data_train[0].shape[0])
        
        #find zero density
        zero_den = (aa < 1e-16).nonzero(as_tuple=True)[0]
        aa[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        logp_x = torch.log(aa)-logp_diff_t0.view(-1)
        
        L2_value2[0][i] = mse(aaa,torch.exp(logp_x.view(-1))) 
        loss = loss  + L2_value2[0][i]*1e4 
        
        
    # compute transport cost efficiency
    transport_cost = partial(trans_loss,func=func,device=device)
    x0 = Sampling(args.num_samples,train_time,0,data_train,0.02,device) 
    logp_diff_t00 = torch.zeros(x0.shape[0], 1).type(torch.float32).to(device)
    g_t00 = logp_diff_t00
    _,_,loss1 = odeint(transport_cost,y0=(x0, g_t00, logp_diff_t00),t = torch.tensor([0, integral_time[-1]]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': 0.1})
    loss = loss + integral_time[-1]*loss1[-1].mean(0)


    if (itr >1):
        if ((itr % 100 == 0) and (itr<=args.niters-400) and (sigma_now>0.02) and (L2_value1.mean()<=0.0003)):
            sigma_now = sigma_now/2

    return loss, loss1, sigma_now, L2_value1, L2_value2
            


def plot_3d(func,data_train,train_time,integral_time,integral_time2,sample_time,args,device):
    viz_samples = 20
    sigma_a = 0.02

    t_list = []#list(reversed(integral_time))#integral_time #np.linspace(5, 0, viz_timesteps)
    #options.update({'t_eval':t_list})
    
    z_t_samples = []
    z_t_data = []
    v = []
    g = []
    t_list2 = [] 
    plot_time = list(reversed(integral_time2))

    with torch.no_grad():
        for i in range(len(integral_time)):

            z_t0 =  data_train[i]

            z_t_data.append(z_t0.cpu().detach().numpy())
            t_list2.append(integral_time[i])
        
        # traj backward
        z_t0 =  Sampling(viz_samples, train_time, len(train_time)-1,data_train,sigma_a,device)
        #z_t0 = z_t0[z_t0[:,2]>1]
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        v_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[0] #True_v(z_t0)
        g_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[1]
        
        v.append(v_t.cpu().detach().numpy())
        g.append(g_t.cpu().detach().numpy())
        z_t_samples.append(z_t0.cpu().detach().numpy())
        t_list.append(plot_time[0])
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})

        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval':plot_time})
        z_t1,_, logp_diff_t1= odesolve(func,y0=(z_t0,g0, logp_diff_t0),options=options)
        for i in range(len(plot_time)-1):
            v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[0] #True_v(z_t0)
            g_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[1]
            
            z_t_samples.append(z_t1[i+1].cpu().detach().numpy())
            g.append(g_t.cpu().detach().numpy())
            v.append(v_t.cpu().detach().numpy())#/20)
            t_list.append(plot_time[i+1])

        aa=5#3
        angle1 = 10#30
        angle2 = 75#30
        trans = 0.8
        trans2 = 0.4
        widths = 0.2 #arrow width
        ratio1 = 0.4

        fig = plt.figure(figsize=(4*2,3*2), dpi=200)
        plt.tight_layout()
        #plt.axis('off')
        plt.margins(0, 0)
        #fig.suptitle(f'{t:.1f}day')
        ax1 = plt.axes(projection ='3d')
        ax1.grid(False)
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.set_zlabel('UMAP3')
        ax1.set_xlim(-2,2)
        ax1.set_ylim(-2,2)
        ax1.set_zlim(-2,2)
        ax1.set_xticks([-2,2])
        ax1.set_yticks([-2,2])
        ax1.set_zticks([-2,2])
        ax1.view_init(elev=angle1, azim=angle2)
        ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.invert_xaxis()
        ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.7, 1]))

        #trans = 0.8#0.2
        #dot_size = 10#5
        line_width = 0.3
        
        
        color_wanted = [np.array([250,187,110])/255,
                        np.array([173,219,136])/255,
                        np.array([250,199,179])/255,
                        np.array([238,68,49])/255,
                        np.array([206,223,239])/255,
                        np.array([3,149,198])/255,
                        np.array([180,180,213])/255,
                        np.array([178,143,237])/255]
        for j in range(viz_samples): #individual traj
            for i in range(len(plot_time)-1):
                ax1.plot([z_t_samples[i][j,0],z_t_samples[i+1][j,0]],
                            [z_t_samples[i][j,1],z_t_samples[i+1][j,1]],
                            [z_t_samples[i][j,2],z_t_samples[i+1][j,2]],
                            linewidth=0.8,color =np.array([132,132,132])/255,zorder=5.2)

                        
        # add inferrred trajecotry
        for i in range(len(sample_time)):
            ax1.scatter(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],z_t_samples[sample_time[i]][:,2],s=aa*5,linewidth=0, color=color_wanted[i],zorder=5.3)

                
        for i in range(len(integral_time)):
            ax1.scatter(z_t_data[i][:,0],z_t_data[i][:,1],z_t_data[i][:,2],s=aa,linewidth=line_width,alpha = 0.7, facecolors='none', edgecolors=color_wanted[i],zorder=5.1)


        # link the traj

        plt.savefig(os.path.join(args.results_dir, f"traj_3d.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
        plt.close()    
            
            
            
