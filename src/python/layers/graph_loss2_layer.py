import math
from torch import nn
from torch.autograd import Function
import torch
import pyDeform

device = None


class GraphLoss2Function(Function):
    @staticmethod
    def forward(ctx, V1, E1, rigidity2, param_id1, param_id2):
        global device
        pid1 = param_id1.tolist()
        pid2 = param_id2.tolist()

        test_V1 = torch.from_numpy(V1.data.cpu().numpy())

        lossD1 = pyDeform.DistanceFieldLoss_forward(test_V1, int(pid2)) * 0.5
        lossR1 = pyDeform.GraphEdgeLoss_forward(test_V1, E1, int(pid1)) * 0.5
        variables = [V1, E1, rigidity2, param_id1, param_id2]
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


class GraphLoss2Layer(nn.Module):
    def __init__(self, V1, F1, graph_V1, graph_E1,
                 V2, F2, graph_V2, graph_E2,
                 rigidity, d=torch.device('cpu')):
        super(GraphLoss2Layer, self).__init__()

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
            return GraphLoss2Function.apply(V1, E1,
                                            self.rigidity2, self.param_id1, self.param_id2)

        return GraphLoss2Function.apply(V2, E2,
                                        self.rigidity2, self.param_id2, self.param_id1)


class GraphLoss2LayerSimple(nn.Module):
    def __init__(self, source_verts, source_faces, source_edges,
                 targ_verts, targ_faces, targ_edges,
                 rigidity, d=torch.device('cpu')):
        super(GraphLoss2LayerSimple, self).__init__()

        global device
        device = d

        self.param_id1 = torch.tensor(
            pyDeform.InitializeDeformTemplate(source_verts, source_faces, 0, 64))

        self.param_id2 = torch.tensor(
            pyDeform.InitializeDeformTemplate(targ_verts, targ_faces, 0, 64))

        pyDeform.NormalizeByTemplate(source_verts, self.param_id1.tolist())
        pyDeform.NormalizeByTemplate(targ_verts, self.param_id2.tolist())

        pyDeform.StoreGraphInformation(
            source_verts, source_edges, self.param_id1.tolist())
        pyDeform.StoreGraphInformation(
            targ_verts, targ_edges, self.param_id2.tolist())
        self.rigidity2 = torch.tensor(rigidity * rigidity)

    def forward(self, V1, E1, V2, E2, direction):
        if direction == 0:
            return GraphLoss2Function.apply(V1, E1,
                                            self.rigidity2, self.param_id1, self.param_id2)

        return GraphLoss2Function.apply(V2, E2,
                                        self.rigidity2, self.param_id2, self.param_id1)

class GraphLoss2LayerMulti(nn.Module):
    def __init__(self, V1, F1, graph_V1, graph_E1,
                 V_targs, F_targs, graph_V_targs, graph_E_targs,
                 rigidity, d=torch.device('cpu')):
        super(GraphLoss2LayerMulti, self).__init__()

        global device
        device = d

        self.param_id1 = torch.tensor(
            pyDeform.InitializeDeformTemplate(V1, F1, 0, 64))

        self.num_targets = len(V_targs)
        param_id_targs = []
        for i in range(self.num_targets):
            param_id_targs.append(torch.tensor(
                pyDeform.InitializeDeformTemplate(V_targs[i], F_targs[i], 0, 64)))
        self.param_id_targs = param_id_targs

        pyDeform.NormalizeByTemplate(graph_V1, self.param_id1.tolist())
        for i in range(self.num_targets):
            pyDeform.NormalizeByTemplate(
                graph_V_targs[i], self.param_id_targs[i].tolist())

        pyDeform.StoreGraphInformation(
            graph_V1, graph_E1, self.param_id1.tolist())
        for i in range(self.num_targets):
            pyDeform.StoreGraphInformation(
                graph_V_targs[i], graph_E_targs[i], self.param_id_targs[i].tolist())
        self.rigidity2 = torch.tensor(rigidity * rigidity)

    def forward(self, V1, E1, V2, E2, index, direction):
        if direction == 0:
            return GraphLoss2Function.apply(V1, E1,
                                            self.rigidity2, self.param_id1, self.param_id_targs[index])

        return GraphLoss2Function.apply(V2, E2,
                                        self.rigidity2, self.param_id_targs[index], self.param_id1)


def Finalize(src_V, src_F, src_E, src_to_graph, graph_V, rigidity, param_id):
    pyDeform.NormalizeByTemplate(src_V, param_id.tolist())
    pyDeform.SolveLinear(src_V, src_F, src_E, src_to_graph, graph_V, rigidity, 1)
    pyDeform.DenormalizeByTemplate(src_V, param_id.tolist())
