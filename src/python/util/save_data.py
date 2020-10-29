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


def save_snapshot_results(V_deformed, GV, V, F, V_targ, F_targ, param_id_targ, output_path):
    GV_copy = GV.clone().cpu()
    V_deformed_copy = V_deformed.detach().clone().cpu()
    pyDeform.DenormalizeByTemplate(GV_copy, param_id_targ)
    pyDeform.DenormalizeByTemplate(V_deformed_copy, param_id_targ)
   
    # Visualize flow.
    flow = V_deformed_copy - GV_copy
    flow_output = output_path[:-4] + "_flow_unmasked.txt"
    np.savetxt(flow_output, flow)
    pyDeform.SaveMesh(output_path[:-4] + "_flow_origin.obj", GV_copy, F)

    # Render the source, target, and result.
    src_output = output_path[:-4] + "_src.png" 
    render_obj(src_output, V, F + 1)
    targ_output = output_path[:-4] + "_targ.png"
    render_obj(targ_output, V_targ, F_targ + 1)
    deformed_output = output_path[:-4] + "_deformed.png"
    render_obj(deformed_output, V_deformed_copy, F + 1)

    # Save obj file.
    pyDeform.SaveMesh(output_path, V_deformed_copy, F)
