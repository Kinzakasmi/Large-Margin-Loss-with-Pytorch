import torch
import torch.nn.functional as F
import numpy as np

def _max_with_relu(a, b):
    return a + F.relu(b - a)

    
def _get_grad(out_, in_):
    grad, *_ = torch.autograd.grad(out_, in_,
                                   grad_outputs=torch.ones_like(out_, dtype=torch.float32),
                                   retain_graph=True)
    return grad.view(in_.shape[0], -1)

class LargeMarginLoss:
    """Large Margin Loss
    A Pytorch Implementation of `Large Margin Deep Networks for Classification`
    Arguments : 
        gamma (float): Desired margin, and distance to boundary above the margin will be clipped.
        top_k (int): Number of top classes to include in the margin loss.
        dist_norm (1, 2, np.inf): Distance to boundary defined on norm
        epslion (float): Small number to avoid division by 0.
        use_approximation (bool):
        agg_type ("all_top_k", "worst_top_k", "avg_top_k"):  If 'worst_top_k'
            only consider the minimum distance to boundary of the top_k classes. If
            'average_top_k' consider average distance to boundary. If 'all_top_k'
            consider all top_k. When top_k = 1, these choices are equivalent.
    """
    def __init__(self, 
                gamma=10000.0,
                top_k=1,
                dist_norm=2,
                epsilon=1e-8,
                use_approximation=True,
                agg_type="all_top_k"):
        
        self.dist_upper = gamma
        
        self.top_k = top_k
        self.dual_norm = {1: np.inf, 2: 2, np.inf: 1}[dist_norm]
        self.eps = epsilon
        
        self.use_approximation = use_approximation
        self.agg_type = agg_type

    def __call__(self, logits, onehot_labels, feature_maps):
        """Getting Large Margin loss
        
        Arguments : 
            logits (Tensor): output of Network before softmax
            onehot_labels (Tensor): One-hot shaped label
            feature_maps (list of Tensor): Target feature maps(Layer of NN) want to enforcing by Large Margin
            
        Returns :
            loss:  Large Margin loss
        """
        prob = F.softmax(logits, dim=1)
        correct_prob = prob * onehot_labels

        correct_prob = torch.sum(correct_prob, dim=1, keepdim=True)
        other_prob = prob * (1.0 - onehot_labels)
        
        if self.top_k > 1:
            topk_prob, _ = other_prob.topk(self.top_k, dim=1)
        else:
            topk_prob, _ = other_prob.max(dim=1, keepdim=True)
        
        # nom of d : f_i(x_k) - f_yk(x_k)
        diff_prob =  topk_prob - correct_prob
        
        loss = torch.empty(0, device=logits.device)
        for feature_map in feature_maps:
            #delta_hl(f_i(x_k)) - delta_hl(f_yk(x_k))
            diff_grad = torch.stack([_get_grad(diff_prob[:, i], feature_map) for i in range(self.top_k)], dim=1)
            #denom of d : dual_norm(delta_hl(f_i(x_k)) - delta_hl(f_yk(x_k)))
            diff_gradnorm = torch.norm(diff_grad, p=self.dual_norm, dim=2)

            if self.use_approximation:
                diff_gradnorm.detach_()
            
            #linear approximation of d :
            dist_to_boundary = diff_prob / (diff_gradnorm + self.eps) #eps to avoid dividing by 0
            
            loss_layer = self.dist_upper + dist_to_boundary
            loss_layer = _max_with_relu(0,loss_layer)

            if self.agg_type == "max_top_k":
                loss_layer, _ = loss_layer.max(dim=1)
            elif self.agg_type == "avg_top_k":
                loss_layer = loss_layer.mean(dim=1)
                        
            loss = torch.cat([loss, loss_layer])
        return loss.mean()