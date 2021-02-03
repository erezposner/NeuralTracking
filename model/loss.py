import sys,os
import torch
import torch.nn as nn
import options as opt
import numpy as np
import math


class DeformLoss(torch.nn.Module):
    def __init__(self, lambda_depth_pred=0, lambda_flow=0, lambda_graph=0, lambda_warp=0, lambda_mask=0, depth_pred_loss_type="L2" ,flow_loss_type="L2"): # L1, L2, MSE (like L2 but w/o sqrt)
        super(DeformLoss, self).__init__()
        self.lambda_depth_pred = lambda_depth_pred
        self.lambda_flow = lambda_flow
        self.lambda_graph = lambda_graph
        self.lambda_warp = lambda_warp
        self.lambda_mask = lambda_mask

        if flow_loss_type == 'RobustL1':
            self.flow_loss = RobustL1()
        elif flow_loss_type == 'L2':
            self.flow_loss = L2()
        else:
            raise Exception("Loss type {} is not defined. Valid losses are 'L1', 'L2' or 'MSE'".format(flow_loss_type))

        if depth_pred_loss_type == 'RobustL1':
            self.depth_pred_loss = torch.nn.L1Loss(reduction='none')
        elif depth_pred_loss_type == 'L2':
            self.depth_pred_loss = torch.nn.MSELoss(reduction='none')
        else:
            raise Exception("Loss type {} is not defined. Valid losses are 'L1', 'L2' or 'MSE'".format(depth_pred_loss_type))

        self.graph_loss = BatchGraphL2()
        self.warp_loss = L2_Warp()


    def forward(self, source_depth_pred, source_depth_gt, target_depth_pred, target_depth_gt,depth_pred_mask,
                flow_gts, flow_preds, flow_masks,
                deformations_gt, deformations_pred, deformations_validity,
                warped_points_gt, warped_points_pred, warped_points_mask,
                valid_solve, num_nodes_vec, mask_pred, mask_gt, valid_pixels,
                evaluate=False):

        d_total = torch.zeros((1), dtype=flow_preds[0].dtype, device=flow_preds[0].device)

        d_depth_pred = None
        if opt.use_depth_pred_loss:
            if len(source_depth_gt) == 1:
                d_depth_pred = self.depth_pred_loss(source_depth_gt[0][:, -1, None, ...], source_depth_pred[0])


                # mask_interpolated = torch.nn.functional.interpolate(input=optical_flow_mask[0][:, 0, None, :, :].float(), size=(d_depth_pred.shape[2], d_depth_pred.shape[3]), mode='nearest')
                # import matplotlib.pyplot as plt
                # plt.imshow((d_depth_pred * mask_interpolated).detach().cpu().numpy()[0][0])
                # z_gt = source_depth_gt[0][:, -1, None, ...].detach().cpu().numpy()
                # norm_val = z_gt[0][0].max()
                # depth_pred_source_Z =( (z_gt[0][0] * 255.0 ) / norm_val).astype(
                #     np.uint8)
                # plt.imshow(depth_pred_source_Z)
                # pred_vis = (source_depth_pred[0][0][0].detach().cpu().numpy()*255.0 / norm_val).astype(
                #     np.uint8)
                # plt.imshow(pred_vis)
                # plt.imshow(source_depth_gt[0][:, -1, None, ...][0][0].detach().cpu().numpy()*depth_pred_mask[0][0][0].detach().cpu().numpy())
                # plt.imshow(source_depth_gt[0][:, -1, None, ...][0][0].detach().cpu().numpy())
                # p = source_depth_gt[0][:, -1, None, ...][0][0].detach().cpu().numpy()*depth_pred_mask[0][0][0].detach().cpu().numpy()
                # plt.clim(min(p.flatten()), max(p.flatten()))
                # plt.show()

                d_depth_pred_src = (d_depth_pred * depth_pred_mask[0])
                d_depth_pred_trg = self.depth_pred_loss(target_depth_gt[0][:, -1, None, ...], target_depth_pred[0])

                d_depth_pred = 0.8*d_depth_pred_src.mean() + 0.2*d_depth_pred_trg.mean()

            elif len(source_depth_gt) > 1:
                d_depth_pred = []
                # for flow_gt, flow_pred, flow_mask in zip(flow_gts, flow_preds, flow_masks):
                for idx in range(len(source_depth_gt)):
                    # It can happen that flow_gt has no valid values for coarser levels.
                    # In that case, that level is not constrained in the batch.
                    f = self.depth_pred_loss(source_depth_gt[idx][:,-1, None,...], source_depth_pred[idx])
                    assert f is not None, f
                    f += (self.depth_pred_loss(target_depth_gt[idx][:,-1, None,...], target_depth_pred[idx])).mean()

                    d_depth_pred.append(f)
                d_depth_pred = sum(d_depth_pred)

            d_total += self.lambda_depth_pred * d_depth_pred

        d_flow = None
        if opt.use_flow_loss:
            if len(flow_gts) == 1:
                d_flow = self.flow_loss(flow_gts[0], flow_preds[0], flow_masks[0])
            elif len(flow_gts) > 1:
                d_flow = []
                for flow_gt, flow_pred, flow_mask in zip(flow_gts, flow_preds, flow_masks):
                    # It can happen that flow_gt has no valid values for coarser levels.
                    # In that case, that level is not constrained in the batch.
                    assert flow_pred is not None, flow_pred
                    f = self.flow_loss(flow_gt, flow_pred, flow_mask)

                    assert f is not None, f
                    d_flow.append(f)
                d_flow = sum(d_flow)

            d_total += self.lambda_flow * d_flow

        d_graph = None
        if opt.use_graph_loss:
            try:
                d_graph = self.graph_loss(deformations_gt, deformations_pred, valid_solve, deformations_validity)
                d_total += self.lambda_graph * d_graph
            except Exception as e:
                print(f'\ncould not use graph_loss due to {e}\n')
                d_graph = torch.tensor([0.0], requires_grad=True)

                pass

        d_warp = None
        if opt.use_warp_loss:
            try:
                d_warp = self.warp_loss(warped_points_gt, warped_points_pred, warped_points_mask)
                d_total += self.lambda_warp * d_warp
            except Exception as e:
                print(f'\ncould not use warp_loss due to {e}\n')
                d_warp = torch.tensor([0.0], requires_grad=True)
                pass
        d_mask = None
        if opt.use_mask_loss:
            d_mask = self.mask_bce_loss(mask_gt, mask_pred, valid_pixels)
            d_total += self.lambda_mask * d_mask

        if evaluate:
            return d_total, d_depth_pred, d_flow, d_graph, d_warp, d_mask
        else:
            return d_total

    @staticmethod
    def epe_2d(flow_gt, flow_pred, flow_mask):
        return EPE_2D(flow_gt, flow_pred, flow_mask)

    @staticmethod
    def epe_warp(points_gt, points_pred, points_mask):
        return EPE_Warp(points_gt, points_pred, points_mask)

    @staticmethod
    def epe_3d(flow_gt, flow_pred, deformations_validity):
        return EPE_3D(flow_gt, flow_pred, deformations_validity)

    @staticmethod
    def mask_bce_distance(mask_gt, mask_pred, valid_pixels):
        return DeformLoss.mask_bce_loss(mask_gt, mask_pred, valid_pixels)
        
    @staticmethod
    def mask_bce_loss(mask_gt, mask_pred, valid_pixels):
        valid_pixels_float = valid_pixels.type(torch.float32)

        # Compute Binary Cross Entropy
        criterion = torch.nn.BCELoss(reduction="none").cuda()
        loss_bce = criterion(mask_pred, mask_gt)
        loss_bce = loss_bce * valid_pixels_float

        # Compute positive and negative masks
        positives_mask = valid_pixels_float * mask_gt
        negatives_mask = valid_pixels_float * (~(mask_gt.clone().type(torch.bool))).type(torch.float32)

        positives_num = torch.sum(positives_mask)
        negatives_num = torch.sum(negatives_mask)
        assert positives_num + negatives_num == torch.sum(valid_pixels_float)

        # Compute relationship between number of negative with respect to number of positive samples
        relative_neg_wrt_pos = negatives_num / positives_num
        
        # Compute weights
        if opt.mask_neg_wrt_pos_weight is not None:
            weights = (opt.mask_neg_wrt_pos_weight * positives_mask) + negatives_mask
        else:
            weights = (relative_neg_wrt_pos * positives_mask) + negatives_mask

        # Finally, apply weights to bce
        loss_bce = weights * loss_bce

        num_valid = torch.sum(valid_pixels_float)
        if num_valid > 0:
            return torch.sum(loss_bce) / torch.sum(valid_pixels_float)
        else:
            return torch.zeros((1), dtype=loss_bce.dtype, device=loss_bce.device)


class BatchGraphL2(nn.Module):
    def __init__(self):
        super(BatchGraphL2, self).__init__()
    
    def forward(self, flow_gt, flow_pred, valid_solve, deformations_validity):
        batch_size = flow_gt.shape[0]

        assert flow_gt.shape[2]   == 3
        assert flow_pred.shape[2] == 3

        assert torch.isfinite(flow_gt).all(), flow_gt

        diff = flow_pred - flow_gt
        diff2 = diff * diff

        deformations_mask = deformations_validity.type(torch.float32)
        deformations_mask = deformations_mask.view(batch_size, -1, 1).repeat(1, 1, 3)
     
        diff2_masked = deformations_mask * diff2
        
        loss = torch.zeros((batch_size), dtype=diff2.dtype, device=diff2.device)
        mask = []
        for i in range(batch_size):
            num_valid_nodes = deformations_validity[i].sum()

            if valid_solve[i] and num_valid_nodes > 0:
                loss[i] = torch.sum(diff2_masked[i]) / num_valid_nodes
                mask.append(i)

        assert torch.isfinite(loss).all()

        if len(mask) == 0:
            return torch.zeros((1), dtype=diff2.dtype, device=flow_gt.device)
        else:
            loss = loss[mask]
            return torch.sum(loss) / len(mask)


class RobustL1(nn.Module):
    """
    Robust training loss for fine-tuning as defined in PWC-Net https://arxiv.org/pdf/1709.02371.pdf
    """
    def __init__(self, epsilon=0.01, q=0.4):
        super(RobustL1, self).__init__()
        self.epsilon = epsilon
        self.q = q
    
    def forward(self, flow_gt, flow_pred, flow_mask):
        batch_size = flow_gt.shape[0]

        assert flow_gt.shape[1]   == 2 or flow_gt.shape[1]   == 3, flow_gt.shape 
        assert flow_pred.shape[1] == 2 or flow_pred.shape[1] == 3, flow_pred.shape 
        assert flow_mask.shape[1] == 2 or flow_mask.shape[1] == 3, flow_mask.shape 


        flow_mask_float = flow_mask[:,0,...] # flow_mask is duplicated across the feature dimension
        flow_mask_float = flow_mask_float.type(torch.float32)

        diff = flow_pred - flow_gt
        lossvalue = torch.abs(diff)
        lossvalue = torch.sum(lossvalue, axis=1) + self.epsilon
        lossvalue = lossvalue**self.q
        lossvalue = lossvalue * flow_mask_float
        assert torch.isfinite(lossvalue).all()

        loss = torch.zeros((batch_size), dtype=lossvalue.dtype, device=lossvalue.device)
        mask = []
        for i in range(batch_size):
            num_valid_i = flow_mask_float[i].sum()
            if num_valid_i > 0:
                loss[i] = torch.sum(lossvalue[i]) / num_valid_i
                mask.append(i)

        if len(mask) == 0:
            return torch.zeros((1), dtype=lossvalue.dtype, device=lossvalue.device)
        else:
            loss = loss[mask]
            return torch.sum(loss) / len(mask)


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    
    def forward(self, flow_gt, flow_pred, flow_mask):
        batch_size = flow_gt.shape[0]

        assert flow_gt.shape[1]   == 2 or flow_gt.shape[1]   == 3, flow_gt.shape 
        assert flow_pred.shape[1] == 2 or flow_pred.shape[1] == 3, flow_pred.shape 
        assert flow_mask.shape[1] == 2 or flow_mask.shape[1] == 3, flow_mask.shape

        assert torch.isfinite(flow_gt).all()
        assert torch.isfinite(flow_pred).all()
        assert torch.isfinite(flow_mask).all()

        flow_mask_float = flow_mask[:,0,...] # flow_mask is duplicated across the feature dimension
        flow_mask_float = flow_mask_float.type(torch.float32)
        
        diff = flow_pred - flow_gt
        lossvalue = torch.norm(diff, p=2, dim=1)
        lossvalue = lossvalue * flow_mask_float
        assert torch.isfinite(lossvalue).all()

        loss = torch.zeros((batch_size), dtype=lossvalue.dtype, device=lossvalue.device)
        mask = []
        for i in range(batch_size):
            num_valid_i = flow_mask_float[i].sum()
            if num_valid_i > 0:
                loss[i] = torch.sum(lossvalue[i]) / num_valid_i
                mask.append(i)

        if len(mask) == 0:
            return torch.zeros((1), dtype=lossvalue.dtype, device=lossvalue.device)
        else:
            loss = loss[mask]
            return torch.sum(loss) / len(mask)


class L2_Warp(nn.Module):
    def __init__(self):
        super(L2_Warp, self).__init__()
    
    def forward(self, points_gt, points_pred, points_mask):
        batch_size = points_gt.shape[0]

        assert torch.isfinite(points_gt).all()
        assert torch.isfinite(points_pred).all()
        assert torch.isfinite(points_mask).all()

        assert points_gt.shape[2]   == 3, points_gt.shape 
        assert points_pred.shape[2] == 3, points_pred.shape 
        assert points_mask.shape[2] == 3, points_mask.shape

        points_mask_float = points_mask[:,:,0] # flow_mask is duplicated across the feature dimension
        points_mask_float = points_mask_float.type(torch.float32)

        diff = points_pred - points_gt
        lossvalue = torch.norm(diff, p=2, dim=2)
        lossvalue = lossvalue * points_mask_float
        assert torch.isfinite(lossvalue).all()

        loss = torch.zeros((batch_size), dtype=points_gt.dtype, device=points_gt.device)
        mask = []
        for i in range(batch_size):
            num_valid_i = points_mask_float[i].sum()
            if num_valid_i > 0:
                loss[i] = torch.sum(lossvalue[i]) / num_valid_i
                mask.append(i)

        if len(mask) == 0:
            return torch.zeros((1), dtype=lossvalue.dtype, device=lossvalue.device)
        else:
            loss = loss[mask]
            return torch.sum(loss) / len(mask)
            

def EPE_2D(flow_gt, flow_pred, flow_mask):
    assert flow_gt.shape[1]   == 2, flow_gt.shape 
    assert flow_pred.shape[1] == 2, flow_pred.shape 
    assert flow_mask.shape[1] == 2, flow_mask.shape

    if torch.sum(flow_mask) == 0:
        return None

    flow_mask = flow_mask[:,0,...] # flow_mask is duplicated across the feature dimension
    flow_mask_float = flow_mask.type(torch.float32)
    flow_mask_bool  = flow_mask.type(torch.bool)
    
    diff = flow_pred - flow_gt
    epe = torch.norm(diff, p=2, dim=1)

    epe = epe * flow_mask_float
    assert torch.isfinite(epe).all()

    return {"sum": epe.sum().item(), "num": flow_mask_float.sum().item(), "raw": epe[flow_mask_bool].cpu().numpy()}


def EPE_Warp(points_gt, points_pred, points_mask):
    assert points_gt.shape[2]   == 3, points_gt.shape 
    assert points_pred.shape[2] == 3, points_pred.shape 
    assert points_mask.shape[2] == 3, points_mask.shape

    if torch.sum(points_mask) == 0:
        return None
    
    points_mask = points_mask[:,:,0] # points_mask is duplicated across the feature dimension
    points_mask_float = points_mask.type(torch.float32)
    points_mask_bool  = points_mask.type(torch.bool)
        
    diff = points_pred - points_gt
    epe = torch.norm(diff, p=2, dim=2)

    epe = epe * points_mask_float
    assert torch.isfinite(epe).all()

    return {"sum": epe.sum().item(), "num": points_mask_float.sum().item(), "raw": epe[points_mask_bool].cpu().numpy()}


def EPE_3D(flow_gt, flow_pred, deformations_validity):
    assert flow_gt.shape[2]   == 3, flow_gt.shape 
    assert flow_pred.shape[2] == 3, flow_pred.shape 
    
    batch_size = flow_gt.shape[0]

    deformations_mask_bool = deformations_validity.type(torch.bool)

    deformations_mask = deformations_validity.type(torch.float32)
    deformations_mask = deformations_mask.view(batch_size, -1)

    diff = flow_pred - flow_gt
    epe = torch.norm(diff, p=2, dim=2)
    epe = deformations_mask * epe 
    assert torch.isfinite(epe).all()

    return {"sum": epe.sum().item(), "num": deformations_validity.sum().item(), "raw": epe[deformations_mask_bool].cpu().numpy()}


def EPE_3D_eval(flow_gt, flow_pred):
    # Used in evaluate.py (no batch size assumed)
    assert len(flow_gt.shape)   == 2, flow_gt.shape
    assert len(flow_pred.shape) == 2, flow_gt.shape
    assert flow_gt.shape[1]   == 3, flow_gt.shape 
    assert flow_pred.shape[1] == 3, flow_pred.shape 

    diff = flow_pred - flow_gt
    epe = np.linalg.norm(diff, axis=1)

    return {"sum": np.sum(epe), "num": flow_gt.shape[0]}
