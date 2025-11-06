import argparse
import os
import sys
from typing import Tuple
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    print("需要依赖 open3d。请先安装: pip install open3d", file=sys.stderr)
    raise


def read_geometry(path: str):
    """
    使用 Open3D 读取 PLY。优先尝试 TriangleMesh，失败时回退为 PointCloud。
    返回 (geometry, kind) 其中 kind ∈ {"mesh", "pcd"}。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 尝试读取为网格
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(np.asarray(mesh.vertices)) > 0:
        mesh.compute_vertex_normals(False)
        return mesh, "mesh"

    # 回退读取为点云
    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(np.asarray(pcd.points)) > 0:
        return pcd, "pcd"

    raise ValueError(f"无法从文件读取顶点: {path}")


def geometry_vertices(geom) -> np.ndarray:
    if isinstance(geom, o3d.geometry.TriangleMesh):
        return np.asarray(geom.vertices)
    elif isinstance(geom, o3d.geometry.PointCloud):
        return np.asarray(geom.points)
    else:
        raise TypeError("不支持的几何类型")


def compute_occlusal_normal(upper_pts: np.ndarray, lower_pts: np.ndarray) -> np.ndarray:
    """
    使用 PCA 对 (upper ∪ lower) 顶点拟合最佳平面，取最小特征值方向为平面法向。
    并将法向调整为从 lower 指向 upper。
    """
    all_pts = np.vstack([upper_pts, lower_pts])
    mean = all_pts.mean(axis=0)
    centered = all_pts - mean
    # 协方差矩阵
    cov = np.cov(centered.T)
    # 特征分解
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, np.argmin(vals)]  # 最小特征值对应的特征向量
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    cu = upper_pts.mean(axis=0)
    cl = lower_pts.mean(axis=0)
    direction_ul = cu - cl  # 下->上
    if np.dot(normal, direction_ul) < 0:
        normal = -normal  # 确保法向从下颌指向上颌

    return normal


def translate_lower(lower_geom, normal: np.ndarray, distance_mm: float):
    """
    将 lower 沿着咬合平面法向量的反方向平移 distance_mm。
    """
    t = (-normal) * float(distance_mm)  # 反方向
    if isinstance(lower_geom, o3d.geometry.TriangleMesh):
        lower_geom.translate(t, relative=True)
    elif isinstance(lower_geom, o3d.geometry.PointCloud):
        pts = np.asarray(lower_geom.points)
        lower_geom.points = o3d.utility.Vector3dVector(pts + t)
    else:
        raise TypeError("不支持的几何类型")
    return lower_geom


def compute_occlusal_frame(upper_pts: np.ndarray, lower_pts: np.ndarray):
    """
    基于法向量 n 构建局部坐标系 (n 上, l 左, f 前)。
    先将点集投影到咬合平面内做 PCA，取平面内的第一主轴为左向初估，
    再用叉乘得到前向，并做一次正交化使其构成右手系。
    """
    n = compute_occlusal_normal(upper_pts, lower_pts)
    all_pts = np.vstack([upper_pts, lower_pts])
    cen = all_pts.mean(axis=0)
    centered = all_pts - cen
    # 投影到平面
    proj = centered - np.outer(centered @ n, n)
    cov2 = np.cov(proj.T)
    vals2, vecs2 = np.linalg.eigh(cov2)
    order = np.argsort(vals2)[::-1]
    l = vecs2[:, order[0]]
    l = l / (np.linalg.norm(l) + 1e-12)
    # 由 l 与 n 得到 f，并正交化
    f = np.cross(l, n)
    f = f / (np.linalg.norm(f) + 1e-12)
    # 重新计算 l，保证严格正交
    l = np.cross(n, f)
    l = l / (np.linalg.norm(l) + 1e-12)
    return n, l, f


def tilt_lower(left_or_front_axis: np.ndarray, lower_geom, tilt_deg: float, use_front_axis: bool = True):
    """
    围绕“前”轴旋转以实现向左侧倾斜。
    采用右手系，选择角度为 -tilt_deg（左侧下沉）。
    """
    axis = left_or_front_axis / (np.linalg.norm(left_or_front_axis) + 1e-12)
    angle_rad = -np.deg2rad(tilt_deg)  # 负角度：左侧下沉
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)
    center = lower_geom.get_center()
    lower_geom.rotate(R, center=center)
    return lower_geom


def main():
    parser = argparse.ArgumentParser(description="估计咬合平面并将 lower.ply 沿法向反方向平移指定毫米数")
    parser.add_argument("--upper", type=str, default="upper.ply", help="上颌 PLY 路径")
    parser.add_argument("--lower", type=str, default="lower.ply", help="下颌 PLY 路径")
    parser.add_argument("--out", type=str, default=None, help="输出下颌 PLY 路径（默认与 lower 同目录，命名为 lower_shifted.ply）")
    parser.add_argument("--distance_mm", type=float, default=-0.14, help="沿咬合平面反方向的平移量，单位毫米")
    parser.add_argument("--tilt_deg", type=float, default=-0.25, help="向左侧倾斜的角度（度），默认 5°")
    args = parser.parse_args()

    upper_geom, _ = read_geometry(args.upper)
    lower_geom, lower_kind = read_geometry(args.lower)

    upper_pts = geometry_vertices(upper_geom)
    lower_pts = geometry_vertices(lower_geom)

    # 局部坐标系：n(上), l(左), f(前)
    n, l, f = compute_occlusal_frame(upper_pts, lower_pts)

    # 平移：沿 n 的反方向
    translate_lower(lower_geom, n, args.distance_mm)

    # 倾斜：围绕“前”轴 f 旋转 -tilt_deg（左侧下沉）
    tilt_lower(f, lower_geom, args.tilt_deg)

    out_path = args.out
    if out_path is None:
        base_dir = os.path.dirname(os.path.abspath(args.lower))
        out_path = os.path.join(base_dir, "lower_shifted.ply")

    ok = False
    if isinstance(lower_geom, o3d.geometry.TriangleMesh):
        ok = o3d.io.write_triangle_mesh(out_path, lower_geom)
    elif isinstance(lower_geom, o3d.geometry.PointCloud):
        ok = o3d.io.write_point_cloud(out_path, lower_geom)

    if not ok:
        raise RuntimeError(f"写出失败: {out_path}")

    print(f"估计的咬合平面法向量(下->上): {n}")
    print(f"已将 lower 沿反方向平移 {args.distance_mm} mm，并围绕前轴左倾 {args.tilt_deg}°")
    print(f"输出文件: {out_path}")


if __name__ == "__main__":
    main()
