# -*- coding: utf-8 -*-
"""
FRD 解析核心 (frd_parser.py)
==============================
仅负责 FRD 文件解析与网格重建, 不包含任何可视化代码(Qt-free):
  - parse_frd():  节点/单元/结果数据集解析(兼容定宽与数字黏连格式)
  - build_mesh(): 重建 PyVista 网格(壳 7/8/9/10, 四面体 3/6)
  - component_array(): 结果分量提取与派生量(MISES/TRESCA/主值/幅值)
  - rotate_camera(): 相机旋转数学(Rodrigues, 供动画使用)
被 frd_plotter.py / tmps_viewer.py 等调用。

出处/引用:
  本工具来自开源项目 TPMS 研究工具链 (Gitee: https://gitee.com/sanreal/TPMS)
  配套教程: B 站 UP 主 Sanreal https://space.bilibili.com/1136187067
  使用本脚本请保留本头部信息并注明出处; 许可证: MIT
"""

import re

import numpy as np
import pyvista as pv


# 兼容定宽(%10d/%12.5E)与数字黏连(1-2.99E+02-1.30E+01)两种写法
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?")


def _nums(s):
    """提取字符串中所有数字(整数/科学计数法), 返回字符串列表"""
    return _NUM_RE.findall(s)


class FrdData:
    def __init__(self):
        self.node_ids = None      # (n,) int
        self.node_xyz = None      # (n, 3)
        self.elems = []           # [(eid, etype, [nodes])]
        self.datasets = []        # [dict(name, comps, time, values{nid: array})]


def parse_frd(path):
    data = FrdData()
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        lines = fp.readlines()

    n = len(lines)
    i = 0
    while i < n:
        ln = lines[i]

        # ---- 节点块(2C 头部与计数在同一行) ----
        if ln.startswith("    2C"):
            i += 1
            ids, xyz = [], []
            while i < n and not lines[i].startswith(" -3"):
                s = lines[i]
                if s.startswith(" -1"):
                    t = _nums(s[3:])
                    if len(t) >= 4:
                        ids.append(int(t[0]))
                        xyz.append([float(t[1]), float(t[2]),
                                    float(t[3])])
                i += 1
            data.node_ids = np.array(ids, dtype=np.int64)
            data.node_xyz = np.array(xyz, dtype=float)
            i += 1                       # 跳过 -3
            continue

        # ---- 单元块(3C 头部与计数在同一行) ----
        if ln.startswith("    3C"):
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                s = lines[i]
                if s.startswith(" -1"):
                    t = _nums(s)
                    if len(t) >= 3:
                        eid = int(t[1])          # t[0] 为块标记 -1
                        etype = int(t[2])
                    conn = []
                    i += 1
                    while i < n and lines[i].startswith(" -2"):
                        conn.extend(int(v) for v in _nums(lines[i])[1:])
                        i += 1
                    data.elems.append((eid, etype, conn))
                    continue
                i += 1
            i += 1
            continue

        # ---- 结果数据集 ----
        if ln.startswith("  100CL"):
            toks = ln.split()
            time = float(toks[2]) if len(toks) > 2 else 0.0
            i += 1
            name, comps, values, ncomp = None, [], {}, 0
            while i < n and not lines[i].startswith(" -3"):
                s = lines[i]
                if s.startswith(" -4"):
                    toks = s.split()
                    name = toks[1]
                    ncomp = int(toks[2])
                elif s.startswith(" -5"):
                    comps.append(s.split()[1])
                elif s.startswith(" -1"):
                    t = _nums(s)
                    if len(t) >= 2:
                        nid = int(t[1])          # t[0] 为块标记 -1
                        values[nid] = np.array(
                            [float(v) for v in t[2:2 + ncomp]])
                i += 1
            if name:
                data.datasets.append(
                    {"name": name, "comps": comps, "time": time,
                     "values": values, "ncomp": ncomp})
            i += 1
            continue

        i += 1

    return data


# ----------------------------------------------------------------------
# 2. 网格重建
# ----------------------------------------------------------------------
WIRE_COLORS = {"黑色": "black", "白色": "white", "灰色": "dimgray"}


_ENV_TEX = None


def _env_texture():
    """生成摄影棚式 equirect 环境贴图(渐变天空 + 柔光箱高光), 缓存单例"""
    global _ENV_TEX
    if _ENV_TEX is not None:
        return _ENV_TEX
    w, h = 512, 256
    v = np.linspace(0.0, 1.0, h)[:, None]
    u = np.linspace(0.0, 1.0, w)[None, :]
    top = np.array([0.90, 0.92, 0.98])
    mid = np.array([0.52, 0.55, 0.62])
    bot = np.array([0.22, 0.21, 0.23])
    img = np.zeros((h, w, 3))
    for i in range(3):
        img[:, :, i] = np.where(
            v < 0.5,
            top[i] + (mid[i] - top[i]) * (v / 0.5),
            mid[i] + (bot[i] - mid[i]) * ((v - 0.5) / 0.5))
    img *= (0.92 + 0.08 * np.sin(u * 2 * np.pi * 2))[..., None]
    for (u0, v0, su, sv, gain) in [
            (0.25, 0.16, 0.10, 0.10, 1.8),
            (0.70, 0.30, 0.16, 0.08, 1.3),
            (0.50, 0.03, 0.35, 0.05, 1.4)]:
        gu = np.exp(-((u - u0) ** 2) / (2 * su ** 2))
        gv = np.exp(-((v - v0) ** 2) / (2 * sv ** 2))
        img += (gain * gu * gv)[:, :, None] * np.array([1.0, 1.0, 1.0])
    img = np.clip(img, 0.0, 1.0)
    _ENV_TEX = pv.Texture((img * 255).astype(np.uint8))
    return _ENV_TEX


def apply_environment(pl, enabled):
    """给渲染器挂环境贴图; 失败静默(PBR 仍可用, 只是反射弱)"""
    if not enabled:
        return
    try:
        pl.add_environment_texture(_env_texture())
        return
    except AttributeError:
        pass
    try:
        pl.renderer.SetEnvironmentTexture(_env_texture(), True)
    except Exception:  # noqa: BLE001
        pass


# 材质预设: 名称 -> (金属度, 粗糙度)
MATERIAL_PRESETS = {
    "塑料": (0.0, 0.45),
    "金属抛光": (1.0, 0.15),
    "金属拉丝": (0.9, 0.35),
    "陶瓷釉面": (0.0, 0.22),
    "橡胶磨砂": (0.0, 0.85),
    "自定义": None,
}


# CCX 单元类型码 -> 类别
SHELL_TYPES = {7: 3, 8: 6, 9: 4, 10: 8}          # 壳: 每单元节点数
SOLID_TYPES = {                                     # 体: 类别 -> 节点数
    "tet": 4, "tet10": 10,
    "hex": 8, "hex20": 20,
    "wedge": 6,
}
_ELEMENT_KINDS = {                                  # etype -> 类别
    3: "tet", 6: "tet10",
    1: "hex", 4: "hex20",
    2: "wedge", 5: "wedge",                         # C3D15 退化用角点
}

# CalculiX 与 VTK 的六面体节点序差异(FreeCAD 文档 FreeCAD->ccx 映射求逆):
# ccx 连接数组经此表取址后得到 VTK 顺序; Abaqus 顺序与 VTK 一致, 恒等映射
_PERM_HEX8_CCX = [7, 4, 5, 6, 3, 0, 1, 2]
_PERM_HEX20_CCX = [7, 4, 5, 6, 3, 0, 1, 2,
                   15, 12, 13, 14, 11, 8, 9, 10,
                   19, 16, 17, 18]


# 各类别对应的 VTK 单元类型(线性与二阶; Abaqus/CCX 节点序与 VTK 兼容)
_VTK_TYPE = {
    "tet": pv.CellType.TETRA,                # 3  C3D4
    "tet10": pv.CellType.QUADRATIC_TETRA,    # 6  C3D10
    "hex": pv.CellType.HEXAHEDRON,           # 1  C3D8
    "hex20": pv.CellType.QUADRATIC_HEXAHEDRON,  # 4  C3D20
    "wedge": pv.CellType.WEDGE,              # 2  C3D6 / 5  C3D15(角点)
}


def build_mesh(data):
    """由 FRD 节点/单元重建 PyVista 网格, 返回 (mesh, displayed_node_ids)。

    支持:
      壳   7/8/9/10 (3/6 节点三角形, 4/8 节点四边形; 高阶取角点显示)
      四面体 3 (C3D4), 6 (C3D10, 全二阶)
      六面体 1 (C3D8), 4 (C3D20/C3D20R, Abaqus 源全二阶, ccx 源角点退化)
      楔形   2 (C3D6), 5 (C3D15, 按角点退化)
    体壳混合时合并为单个 UnstructuredGrid(壳面以 POLYGON 并入);
    仅壳时返回 PolyData。
    """
    if data.node_xyz is None or len(data.node_ids) == 0:
        raise ValueError("FRD 文件缺少节点数据")
    nid2row = {int(nid): r for r, nid in enumerate(data.node_ids)}

    faces, used, solids = [], set(), []
    for _eid, etype, conn in data.elems:
        if etype in SHELL_TYPES:
            k = 3 if etype == 8 else (4 if etype == 10 else SHELL_TYPES[etype])
            faces.append(conn[:k])
            used.update(conn[:k])
            continue
        kind = _ELEMENT_KINDS.get(etype)
        if kind is None:
            continue
        nnode = SOLID_TYPES[kind]
        if len(conn) < nnode:
            continue
        solids.append((kind, conn[:nnode]))
        used.update(conn[:nnode])

    if not used:
        raise ValueError("FRD 单元类型暂不支持"
                         "(支持壳 7/8/9/10, 四面体 3/6, 六面体 1/4, 楔形 2/5)")

    used = sorted(used)
    remap = {nid: r for r, nid in enumerate(used)}
    verts = data.node_xyz[[nid2row[n] for n in used]]

    if solids:
        # 自动检测六面体节点序约定:
        # CalculiX 的 C3D8/C3D20 连接序与 Abaqus/VTK 不同,
        # 采样有向体积(角点 0,1,2,4 构成四面体), 多数为负则按 ccx 约定重排
        vols = []
        for kind, conn in solids[:200]:
            if kind in ("hex", "hex20"):
                r = [remap[c] for c in conn]
                a, b, c, d = (verts[r[i]] for i in (0, 1, 2, 4))
                vols.append(float(np.dot(b - a, np.cross(c - a, d - a))))
        ccx_order = bool(vols) and float(np.mean(vols)) < 0.0

        # VTK9 混合单元格式: offsets + 展平 connectivity
        # 节点序依据(官方文档核实):
        #   C3D10: Abaqus/ccx/VTK 中点序完全一致 -> 保留全二阶;
        #   C3D20: 仅 Abaqus 源与 VTK 一致(底/顶/竖分组) -> Abaqus 源保留
        #          全二阶; ccx 源中点序不同 -> 角点退化 + 角点置换
        sizes, conns, ctypes = [], [], []
        for kind, conn in solids:
            r = [remap[c] for c in conn]
            if kind == "hex20" and ccx_order:
                r, kind = r[:8], "hex"
            elif kind == "tet10" and ccx_order:
                r, kind = r[:4], "tet"
            if ccx_order and kind == "hex":
                r = [r[i] for i in _PERM_HEX8_CCX]
            sizes.append(len(r))
            conns.extend(r)
            ctypes.append(_VTK_TYPE[kind])
        for f in faces:                     # 壳面并入(混合模型)
            r = [remap[c] for c in f]
            sizes.append(len(r))
            conns.extend(r)
            ctypes.append(pv.CellType.POLYGON)
        offsets = np.cumsum(sizes, dtype=np.int64)
        connectivity = np.array(conns, dtype=np.int64)
        ctypes_arr = np.array(ctypes, dtype=np.uint8)
        verts = np.asarray(verts, dtype=float)
        mesh = None
        try:
            # 新版 PyVista: VTK9 (offsets, connectivity, types, points)
            mesh = pv.UnstructuredGrid(offsets, connectivity,
                                       ctypes_arr, verts)
        except (TypeError, ValueError):
            if len(set(sizes)) == 1:
                try:
                    # 旧版: 等尺寸 2D 数组 (ncells, n+1)
                    n0 = sizes[0]
                    padded = connectivity.reshape(-1, n0)
                    cells2d = np.hstack(
                        [np.full((len(sizes), 1), n0, np.int64), padded])
                    mesh = pv.UnstructuredGrid(cells2d, ctypes_arr, verts)
                except (TypeError, ValueError):
                    mesh = None
        if mesh is None:
            # 兜底: vtk 原生构造(混合尺寸, 全版本可用)
            import vtk
            from vtk.util.numpy_support import (numpy_to_vtk,
                                                numpy_to_vtkIdTypeArray)
            vtk_pts = vtk.vtkPoints()
            vtk_pts.SetData(numpy_to_vtk(verts))
            cell_arr = vtk.vtkCellArray()
            cell_arr.SetData(numpy_to_vtkIdTypeArray(offsets),
                             numpy_to_vtkIdTypeArray(connectivity))
            ug = vtk.vtkUnstructuredGrid()
            ug.SetPoints(vtk_pts)
            ug.SetCells(numpy_to_vtk(
                ctypes_arr, array_type=vtk.VTK_UNSIGNED_CHAR), cell_arr)
            mesh = pv.UnstructuredGrid(ug)
    else:
        farr = np.array([[remap[c] for c in f] for f in faces],
                        dtype=np.int64)
        pad = np.full((len(farr), 1), farr.shape[1], dtype=np.int64)
        mesh = pv.PolyData(verts, np.hstack([pad, farr]))

    return mesh, np.array(used, dtype=np.int64)


# 2.5 相机旋转(手动 Rodrigues, 与 PyVista 版本无关)
# ----------------------------------------------------------------------
def _rotate_about(center, vec, axis, deg):
    k = axis / np.linalg.norm(axis)
    v = np.asarray(vec, dtype=float)
    th = np.deg2rad(deg)
    return (v * np.cos(th) + np.cross(k, v) * np.sin(th)
            + k * np.dot(k, v) * (1 - np.cos(th)))


def _viewup(cam):
    """兼容不同 PyVista 版本的视向上向量获取"""
    if hasattr(cam, "viewup"):
        return np.asarray(cam.viewup, dtype=float)
    if hasattr(cam, "view_up"):
        return np.asarray(cam.view_up, dtype=float)
    inner = getattr(cam, "_camera", cam)
    return np.asarray(inner.GetViewUp(), dtype=float)


def _set_viewup(cam, v):
    """兼容不同 PyVista 版本的视向上向量设置"""
    v = np.asarray(v, dtype=float)
    if hasattr(cam, "viewup"):
        cam.viewup = v
    elif hasattr(cam, "view_up"):
        cam.view_up = v
    else:
        inner = getattr(cam, "_camera", cam)
        inner.SetViewUp(float(v[0]), float(v[1]), float(v[2]))


def rotate_camera(pl, d_az=0.0, d_el=0.0, d_roll=0.0):
    cam = pl.camera
    fp = np.asarray(cam.focal_point, dtype=float)
    if d_az:
        pos = np.asarray(cam.position, dtype=float)
        cam.position = fp + _rotate_about(fp, pos - fp, _viewup(cam), d_az)
    if d_el:
        pos = np.asarray(cam.position, dtype=float)
        axis = np.cross(fp - pos, _viewup(cam))
        cam.position = fp + _rotate_about(fp, pos - fp, axis, d_el)
    if d_roll:
        pos = np.asarray(cam.position, dtype=float)
        view = fp - pos
        _set_viewup(cam, _rotate_about(fp, _viewup(cam), view, d_roll))


# ----------------------------------------------------------------------


# 3. 标量场计算(含派生量)
# ----------------------------------------------------------------------
def components_for(ds):
    name, comps, ncomp = ds["name"], ds["comps"], ds["ncomp"]
    out = list(comps)
    if ncomp == 6 and name in ("STRESS", "ExtrapS"):
        out += ["MISES", "TRESCA", "PRINCIPAL-MAX", "PRINCIPAL-MID",
                "PRINCIPAL-MIN"]
    elif ncomp == 6 and name in ("TOSTRAIN", "ExtrapE"):
        out += ["MISES", "PRINCIPAL-MAX", "PRINCIPAL-MID", "PRINCIPAL-MIN"]
    elif ncomp == 6:
        out += ["MISES"]
    elif ncomp >= 3 and name in ("DISP", "UR", "RF", "RM"):
        out += ["MAGNITUDE"]
    return out


def component_array(ds, comp, node_ids):
    n = len(node_ids)
    M = np.full((n, ds["ncomp"]), np.nan)
    id2row = {int(nid): r for r, nid in enumerate(node_ids)}
    for nid, v in ds["values"].items():
        r = id2row.get(int(nid))
        if r is not None and len(v) >= 1:
            # 实际分量数可能少于声明值(如 DISP 声明 4 维但只写 3 个),
            # 按实际长度填充, 缺失分量保持 NaN
            k = min(len(v), ds["ncomp"])
            M[r, :k] = v[:k]

    # 声明的 ALL 分量(第 4 维)实际没有数据, 用前三分量算幅值
    if comp == "ALL" and ds["ncomp"] >= 3:
        return np.sqrt((M[:, :3] ** 2).sum(axis=1))

    if comp in ds["comps"]:
        return M[:, ds["comps"].index(comp)]

    if ds["ncomp"] == 6:
        s11, s22, s33 = M[:, 0], M[:, 1], M[:, 2]
        s12, s23, s13 = M[:, 3], M[:, 4], M[:, 5]
        if comp == "MISES":
            return np.sqrt(
                0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2
                       + (s33 - s11) ** 2)
                + 3.0 * (s12 ** 2 + s23 ** 2 + s13 ** 2))
        if comp in ("PRINCIPAL-MAX", "PRINCIPAL-MID", "PRINCIPAL-MIN",
                    "TRESCA"):
            T = np.zeros((n, 3, 3))
            T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s11, s22, s33
            T[:, 0, 1] = T[:, 1, 0] = s12
            T[:, 1, 2] = T[:, 2, 1] = s23
            T[:, 0, 2] = T[:, 2, 0] = s13
            w = np.linalg.eigvalsh(T)          # 升序
            if comp == "PRINCIPAL-MAX":
                return w[:, 2]
            if comp == "PRINCIPAL-MID":
                return w[:, 1]
            if comp == "PRINCIPAL-MIN":
                return w[:, 0]
            return w[:, 2] - w[:, 0]           # TRESCA

    if comp == "MAGNITUDE" and ds["ncomp"] >= 3:
        return np.sqrt((M[:, :3] ** 2).sum(axis=1))

    raise KeyError(f"未知分量: {comp}")


# ----------------------------------------------------------------------
# 4. 主窗口
# ----------------------------------------------------------------------


WIRE_COLORS = {"黑色": "black", "白色": "white", "灰色": "dimgray"}


MATERIAL_PRESETS = {
    "塑料": (0.0, 0.45),
    "金属抛光": (1.0, 0.15),
    "金属拉丝": (0.9, 0.35),
    "陶瓷釉面": (0.0, 0.22),
    "橡胶磨砂": (0.0, 0.85),
    "自定义": None,
}

