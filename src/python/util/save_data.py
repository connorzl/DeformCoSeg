import os
import sys
import subprocess
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'build')))

import torch
import pyDeform
import numpy as np

def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))


def render_obj(out, v, f, delete_img=False, flat_shading=True):
    tmp_obj = out.replace('.png', '.obj')

    export_obj(tmp_obj, v, f)

    if flat_shading:
        cmd = 'RenderShape -0 %s %s 600 600 > /dev/null' % (tmp_obj, out)
    else:
        cmd = 'RenderShape %s %s 600 600 > /dev/null' % (tmp_obj, out)

    subprocess.run(cmd, shell=True)
    
    cmd = 'rm -rf %s' % (tmp_obj)
    subprocess.run(cmd, shell=True)


def save_results(V, F, E, V_targ, F_targ, func, param_id_src, param_id_targ, \
        output_path, device, latent_code=None):
    V_copy = V.clone()
    src_to_src = torch.from_numpy(np.array([i for i in range(V_copy.shape[0])]).astype('int32'))

    pyDeform.NormalizeByTemplate(V_copy, param_id_src)
    V_copy = V_copy.to(device)
    if latent_code == None:
        V_deformed = func.forward(V_copy)
    else:
        V_deformed = func.forward(V_copy, latent_code)
    V_deformed = torch.from_numpy(V_deformed.detach().cpu().numpy())

    V_origin = V.clone()
    pyDeform.SolveLinear(V_origin, F, E, src_to_src, V_deformed, 1, 1)
    pyDeform.DenormalizeByTemplate(V_origin, param_id_targ)
    
    # Render the source, target, and result.
    src_output = output_path[:-4] + "_src.png" 
    render_obj(src_output, V, F + 1)
    targ_output = output_path[:-4] + "_targ.png"
    render_obj(targ_output, V_targ, F_targ + 1)
    deformed_output = output_path[:-4] + "_deformed.png"
    render_obj(deformed_output, V_origin, F + 1)
    
    # Save obj file.
    pyDeform.SaveMesh(output_path, V_origin, F)


def save_seg_results(V_parts, V, F, E, V_targ, F_targ, deformers, param_id_src, param_id_targ, \
        output_path, device, latent_code=None):
    V_copy = V.clone()
    V_origin = V.clone()
    src_to_src = torch.from_numpy(np.array([i for i in range(V_copy.shape[0])]).astype('int32'))

    pyDeform.NormalizeByTemplate(V_copy, param_id_src)
    deformed_parts = []
    for i, part in enumerate(V_parts):
        part_device = part.to(device)
        if latent_code == None:
            deformed_parts.append(deformers[i].forward(part_device))
        else:
            deformed_parts.append(deformers[i].forward(part_device, latent_code))
    V_deformed = torch.cat(deformed_parts, dim=0)
    V_deformed = torch.from_numpy(V_deformed.detach().cpu().numpy())

    pyDeform.SolveLinear(V_origin, F, E, src_to_src, V_deformed, 1, 1)
    pyDeform.DenormalizeByTemplate(V_origin, param_id_targ)
    
    # Render the source, target, and result.
    src_output = output_path[:-4] + "_src.png" 
    render_obj(src_output, V, F + 1)
    targ_output = output_path[:-4] + "_targ.png"
    render_obj(targ_output, V_targ, F_targ + 1)
    deformed_output = output_path[:-4] + "_deformed.png"
    render_obj(deformed_output, V_origin, F + 1)
    
    # Save obj file.
    pyDeform.SaveMesh(output_path, V_origin, F)


def save_snapshot_results(V, V_deformed, F, E, V_targ, F_targ, param_id_src, param_id_targ, output_path):
    V_copy = V.clone()
    V_origin = V.clone()
    V_deformed = torch.from_numpy(V_deformed.detach().cpu().numpy())
    src_to_src = torch.from_numpy(np.array([i for i in range(V_copy.shape[0])]).astype('int32'))
    
    pyDeform.NormalizeByTemplate(V_copy, param_id_src)
    pyDeform.SolveLinear(V_origin, F, E, src_to_src, V_deformed, 1, 1)
    pyDeform.DenormalizeByTemplate(V_origin, param_id_targ)
    
    # Render the source, target, and result.
    src_output = output_path[:-4] + "_src.png" 
    render_obj(src_output, V, F + 1)
    targ_output = output_path[:-4] + "_targ.png"
    render_obj(targ_output, V_targ, F_targ + 1)
    deformed_output = output_path[:-4] + "_deformed.png"
    render_obj(deformed_output, V_origin, F + 1)

    # Save obj file.
    pyDeform.SaveMesh(output_path, V_origin, F)

