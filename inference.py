#!/usr/bin/env python
"""
单文件推理脚本
python inference.py --upper data/upper.ply --lower data/lower.ply \
                    --weights checkpoints/best.pth --out_dir results/
"""
import argparse, os, torch, trimesh, numpy as np
from geo_encoder  import FNOGeoEncoder
from physics_core import PhysicsCore
from losses       import compute_loss   # 仅用于日志
from pytorch3d.transforms import so3_exponential_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(cp_file):
    ckpt = torch.load(cp_file, map_location=device)
    enc_up  = FNOGeoEncoder().to(device)
    enc_low = FNOGeoEncoder().to(device)
    core    = PhysicsCore().to(device)
    enc_up.load_state_dict(ckpt['enc_up'])
    enc_low.load_state_dict(ckpt['enc_low'])
    core.load_state_dict(ckpt['core'])
    return enc_up.eval(), enc_low.eval(), core.eval()

@torch.no_grad()
def infer(V_up, F_up, V_low, enc_up, enc_low, core, n_collo=8000):
    core.set_V_up(V_up)
    z_up  = enc_up(V_up)
    z_low = enc_low(V_low)
    # 采样接触点
    collo = V_up[torch.randperm(V_up.shape[0])[:n_collo]]
    theta, t = core(collo, z_up, z_low)
    # 全局变换
    R = so3_exponential_map(theta.unsqueeze(0)).squeeze(0)
    mandible_v = V_low @ R.T + t
    T = torch.eye(4, device=device)
    T[:3,:3] = R; T[:3,3] = t
    return mandible_v, T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper', required=True, help='上颌 PLY')
    parser.add_argument('--lower', required=True, help='下颌 PLY')
    parser.add_argument('--weights', required=True, help='best.pth')
    parser.add_argument('--out_dir', default='results', help='输出文件夹')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 读网格
    upper = trimesh.load(args.upper)
    lower = trimesh.load(args.lower)
    V_up  = torch.tensor(upper.vertices, dtype=torch.float32, device=device)
    F_up  = torch.tensor(upper.faces,    dtype=torch.long,   device=device)
    V_low = torch.tensor(lower.vertices, dtype=torch.float32, device=device)

    # 2. 加载模型
    enc_up, enc_low, core = load_models(args.weights)

    # 3. 推理
    mandible_v, T = infer(V_up, F_up, V_low, enc_up, enc_low, core)

    # 4. 保存
    out_mesh = trimesh.Trimesh(vertices=mandible_v.cpu().numpy(), faces=lower.faces)
    out_mesh.export(os.path.join(args.out_dir, 'lower_bite.ply'))
    np.savetxt(os.path.join(args.out_dir, 'T_matrix.csv'), T.cpu().numpy(), delimiter=',')
    print(f'结果已写入 {args.out_dir}/lower_bite.ply  &  T_matrix.csv')
    print('T=')
    print(T.cpu().numpy())

if __name__ == '__main__':
    main()