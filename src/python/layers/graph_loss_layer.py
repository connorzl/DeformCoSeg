import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform
import time
device = None


class GraphLossFunction(Function):
    @staticmethod
    def forward(ctx, V1, E1, rigidity2, param_id1, param_id2):
        global device
        test_V1 = torch.from_numpy(V1.data.cpu().numpy())
            
        lossD1 = pyDeform.DistanceFieldLoss_forward(test_V1, int(param_id2)) * 0.5
        lossR1 = pyDeform.GraphEdgeLoss_forward(test_V1, E1, int(param_id1)) * 0.5
        variables = [V1, E1, rigidity2, torch.tensor(param_id1), torch.tensor(param_id2)]
        ctx.save_for_backward(*variables)
        return (lossD1.sum() + lossR1.sum() * rigidity2.tolist()).to(device)

    @staticmethod
    def backward(ctx, grad_h):
        global device
        V1 = ctx.saved_variables[0]
        E1 = ctx.saved_variables[1]
        rigidity2 = ctx.saved_variables[2]
        param_id1 = ctx.saved_variables[3].tolist()
        param_id2 = ctx.saved_variables[4].tolist()

        test_V1 = torch.from_numpy(V1.data.cpu().numpy())

        lossD1_gradient = pyDeform.DistanceFieldLoss_backward(
            test_V1, param_id2)
        lossR1_gradient = pyDeform.GraphEdgeLoss_backward(
            test_V1, E1, param_id1)

        return (grad_h*(lossD1_gradient + lossR1_gradient*rigidity2.tolist()).to(device)),\
            None, None, None, None

class RigidityLossFunction(Function):
    @staticmethod
    def forward(ctx, V1, E1, rigidity2, param_id1):
        global device
        test_V1 = torch.from_numpy(V1.data.cpu().numpy())
            
        lossR1 = pyDeform.GraphEdgeLoss_forward(test_V1, E1, int(param_id1)) * 0.5
        variables = [V1, E1, rigidity2, torch.tensor(param_id1)]
        ctx.save_for_backward(*variables)
        return (lossR1.sum() * rigidity2.tolist()).to(device)

    @staticmethod
    def backward(ctx, grad_h):
        global device
        V1 = ctx.saved_variables[0]
        E1 = ctx.saved_variables[1]
        rigidity2 = ctx.saved_variables[2]
        param_id1 = ctx.saved_variables[3].tolist()

        test_V1 = torch.from_numpy(V1.data.cpu().numpy())

        lossR1_gradient = pyDeform.GraphEdgeLoss_backward(
            test_V1, E1, param_id1)

        return (grad_h*(lossR1_gradient*rigidity2.tolist()).to(device)),\
            None, None, None

class GraphLossLayer(nn.Module):
    def __init__(self, V1, F1, graph_V1, graph_E1,
                 V2, F2, graph_V2, graph_E2,
                 rigidity, d=torch.device('cpu')):
        super(GraphLossLayer, self).__init__()

        global device
        device = d

        self.param_id1 = torch.tensor(
            pyDeform.InitializeDeformTemplate(V1, F1, 0, 64))

        self.param_id2 = torch.tensor(
            pyDeform.InitializeDeformTemplate(V2, F2, 0, 64))

        pyDeform.NormalizeByTemplate(graph_V1, self.param_id1.tolist())
        pyDeform.NormalizeByTemplate(graph_V2, self.param_id2.tolist())

        pyDeform.StoreGraphInformation(
            graph_V1, graph_E1, self.param_id1.tolist())
        pyDeform.StoreGraphInformation(
            graph_V2, graph_E2, self.param_id2.tolist())
        self.rigidity2 = torch.tensor(rigidity * rigidity)

    def forward(self, V1, E1, V2, E2, direction):
        if direction == 0:
            return GraphLossFunction.apply(V1, E1,
                                            self.rigidity2, self.param_id1, self.param_id2)

        return GraphLossFunction.apply(V2, E2,
                                        self.rigidity2, self.param_id2, self.param_id1)


class GraphLossLayerBatch(nn.Module):
    def __init__(self, rigidity, d=torch.device('cpu')):
        super(GraphLossLayerBatch, self).__init__()

        global device
        device = d
        self.rigidity2 = torch.tensor(rigidity * rigidity)
    
    def forward(self, V1, E1, V2, E2, src_param_id, tar_param_id, direction):
        if direction == 0:
            return GraphLossFunction.apply(V1, E1,
                                           self.rigidity2, src_param_id, tar_param_id)
        return GraphLossFunction.apply(V2, E2,
                                        self.rigidity2, tar_param_id, src_param_id)

class IntermediateLossLayer(nn.Module):
    def __init__(self, rigidity):
        super(IntermediateLossLayer, self).__init__()

        self.rigidity2 = torch.tensor(rigidity * rigidity)

    def forward(self, V1, E1, src_param_id):
        return RigidityLossFunction.apply(V1, E1, self.rigidity2, src_param_id)


def Finalize(src_V, src_F, src_E, src_to_graph, graph_V, rigidity, param_id):
    pyDeform.NormalizeByTemplate(src_V, param_id.tolist())
    pyDeform.SolveLinear(src_V, src_F, src_E, src_to_graph, graph_V, rigidity, 1)
    pyDeform.DenormalizeByTemplate(src_V, param_id.tolist())
