# -*- coding: utf-8 -*-
"""
FRD 结果显示器 (frd_plotter.py)
=================================
在 implicit_plotter.py 界面框架基础上构建的 FRD 结果查看工具:
  - 打开 odb2frd.py 导出的 CalculiX FRD 文件
  - 选项树: Geometry(几何云图) + Field outputs(结果云图)
  - 渲染: PBR 材质预设 / 环境反射 IBL / EDL / 离散云图 / 变形显示
解析核心复用同目录的 frd_parser.py 模块(纯解析, 无 Qt 依赖)。
依赖:
  pip install pyvista pyvistaqt PySide6 numpy
运行:
  python frd_plotter.py [xxx.frd]

出处/引用:
  本工具来自开源项目 TPMS 研究工具链 (Gitee: https://gitee.com/sanreal/TPMS)
  配套教程: B 站 UP 主Sanreal https://space.bilibili.com/1136187067
  使用本脚本请保留本头部信息并注明出处; 许可证: MIT
"""

import os
import sys

os.environ.setdefault("QT_API", "pyside6")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets, QtCore, QtGui

import frd_parser  # FRD 解析核心(parse_frd / build_mesh / component_array ...)
import frd_convert  # odb -> FRD 转换

APP_TITLE = "FRD 结果显示器"

# ----------------------------------------------------------------------
# 截断版 rainbow: 红->黄->绿->青->蓝, 去掉紫端
# ----------------------------------------------------------------------
_RAINBOW_CUT = None


# ----------------------------------------------------------------------
# 程序化环境贴图(IBL): PBR 材质的环境反射来源
# ----------------------------------------------------------------------
_ENV_TEX = None


def _env_texture():
    """生成摄影棚式 equirect 环境贴图(渐变天空 + 柔光箱高光), 缓存单例"""
    global _ENV_TEX
    if _ENV_TEX is not None:
        return _ENV_TEX
    w, h = 512, 256
    v = np.linspace(0.0, 1.0, h)[:, None]          # 0=顶部(天空)
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


def _cmap(name):
    """色图名 -> 可用对象; 'rainbow' 返回截断紫端的版本"""
    global _RAINBOW_CUT
    if name != "rainbow":
        return name
    if _RAINBOW_CUT is None:
        from matplotlib import colormaps
        # jet: 深蓝(低)->青->绿->黄->红(高), 无紫色, 经典彩虹方向
        _RAINBOW_CUT = colormaps["jet"]
    return _RAINBOW_CUT


_ICON_CACHE = {}


def _icon(kind):
    """程序生成 16x16 图标(无外部图片文件); kind 决定颜色与形状"""
    if kind in _ICON_CACHE:
        return _ICON_CACHE[kind]

    styles = {
        "geom":   ("#8a8f98", "box"),       # Geometry 根: 立方体
        "root":   ("#4a7ebb", "bars"),      # Field outputs 根: 柱状图
        "time":   ("#b89a4e", "clock"),      # 时间子节点: 时钟
        "leaf":   ("#666666", "dot"),       # 分量叶子: 小圆点
        "DISP":   ("#3a7bd5", "circle"),    # 位移类: 蓝圆
        "UR":     ("#3a7bd5", "circle"),
        "RF":     ("#7a5bd6", "circle"),    # 反力类: 紫圆
        "RM":     ("#7a5bd6", "circle"),
        "STRESS": ("#d54a3a", "rect"),      # 应力: 红方
        "ExtrapS": ("#d54a3a", "rect"),
        "TOSTRAIN": ("#3aa655", "diamond"),  # 应变类: 绿菱
        "ExtrapE": ("#3aa655", "diamond"),
        "PE":     ("#3aa655", "diamond"),
        "PEEQ":   ("#e08a2e", "triangle"),  # 标量类: 橙三角
        "PEMAG":  ("#e08a2e", "triangle"),
        "NT11":   ("#2eb8c0", "circle"),    # 温度: 青圆
    }
    color, shape = styles.get(kind, ("#999999", "circle"))

    pm = QtGui.QPixmap(16, 16)
    pm.fill(QtCore.Qt.transparent)
    pt = QtGui.QPainter(pm)
    pt.setRenderHint(QtGui.QPainter.Antialiasing)
    pt.setBrush(QtGui.QBrush(QtGui.QColor(color)))
    pt.setPen(QtGui.QPen(QtGui.QColor("#333333"), 1))
    if shape == "circle":
        pt.drawEllipse(2, 2, 12, 12)
    elif shape == "rect":
        pt.drawRect(2, 2, 12, 12)
    elif shape == "diamond":
        pts = [QtCore.QPoint(8, 1), QtCore.QPoint(15, 8),
               QtCore.QPoint(8, 15), QtCore.QPoint(1, 8)]
        pt.drawPolygon(pts)
    elif shape == "triangle":
        pts = [QtCore.QPoint(8, 2), QtCore.QPoint(14, 14),
               QtCore.QPoint(2, 14)]
        pt.drawPolygon(pts)
    elif shape == "dot":
        pt.drawEllipse(5, 5, 6, 6)
    elif shape == "clock":
        pt.drawEllipse(1, 1, 14, 14)
        pt.setPen(QtGui.QPen(QtGui.QColor("#333333"), 2))
        pt.drawLine(QtCore.QPoint(8, 8), QtCore.QPoint(8, 4))
        pt.drawLine(QtCore.QPoint(8, 8), QtCore.QPoint(11, 9))
    elif shape == "box":
        pt.drawRect(2, 2, 12, 12)
        pt.setPen(QtGui.QPen(QtGui.QColor("#333333"), 1))
        pt.drawRect(5, 5, 9, 9)              # 内框营造立方体感
    elif shape == "bars":
        for i, h in enumerate((8, 12, 6)):
            pt.drawRect(2 + i * 5, 14 - h, 4, h)
    pt.end()

    icon = QtGui.QIcon(pm)
    _ICON_CACHE[kind] = icon
    return icon


def _pixmap_to_np(pixmap):
    """QPixmap -> RGB numpy 数组(供 imageio 写帧)"""
    img = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    bits = img.bits()
    try:
        bits.setsize(img.sizeInBytes())
    except (AttributeError, TypeError):
        pass
    arr = np.frombuffer(bits, np.uint8).reshape(h, w, 4)
    return arr[:, :, :3].copy()


def _mp4_available():
    """检测 imageio 的 FFMPEG 插件(MP4 输出依赖)"""
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


class FrdOptionsDialog(QtWidgets.QDialog):
    """FRD 转换场选择对话框"""

    FIELDS = [("U", "DISP displacement"),
              ("S", "STRESS stress"),
              ("E", "STRAIN strain"),
              ("PE", "plastic strain"),
              ("PEEQ", "equiv. plastic strain"),
              ("PEMAG", "plastic strain magnitude"),
              ("RF", "reaction force"),
              ("RM", "reaction moment"),
              ("NT11", "temperature")]

    SETTINGS = QtCore.QSettings("TPMSTools", "TMPSViewer")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FRD 转换选项")
        self.setMinimumWidth(320)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("选择要转换的场(默认 DISP 与 S):"))

        self.boxes = {}
        for key, desc in self.FIELDS:
            cb = QtWidgets.QCheckBox(f"{key}   ({desc})")
            self.boxes[key] = cb
            lay.addWidget(cb)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        prev = str(self.SETTINGS.value("frd/fields", "U,S"))
        self.set_selected(prev.split(","))

    def selected(self):
        return [k for k, _ in self.FIELDS if self.boxes[k].isChecked()]

    def set_selected(self, keys):
        keys = set(keys)
        for k in self.FIELDS:
            self.boxes[k[0]].setChecked(k[0] in keys)

    def accept(self):
        self.SETTINGS.setValue("frd/fields", ",".join(self.selected()))
        self.SETTINGS.sync()
        super().accept()


class LegendWidget(QtWidgets.QWidget):
    """可拖动的云图图例部件(Abaqus 风格自绘)。

    半透明底 + 标题 + 离散色带 + 科学计数标签; 左键拖动换位。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = ""
        self._vmin = 0.0
        self._vmax = 1.0
        self._levels = 11
        self._cmap_obj = None
        self._drag_pos = None
        self.setToolTip("左键拖动移动图例位置")
        self.setMinimumSize(60, 40)

    def set_data(self, title, vmin, vmax, levels, cmap_obj):
        self._title = title
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._levels = max(2, int(levels))
        self._cmap_obj = cmap_obj
        self.updateGeometry()
        self.adjustSize()
        self.update()

    def sizeHint(self):
        from PySide6 import QtGui
        fm = self.fontMetrics()
        sample = "%+.3e" % max(abs(self._vmin), abs(self._vmax), 1e-30)
        w = 10 + 18 + 6 + fm.horizontalAdvance(sample) + 10
        h = 6 + fm.height() + 4 + self._levels * 14 + 8
        return QtCore.QSize(w, h)

    def paintEvent(self, _ev):
        from PySide6 import QtGui
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 190))
        fm = self.fontMetrics()
        x0, y = 10, 6
        # 标题(加粗)
        f = self.font()
        f.setBold(True)
        p.setFont(f)
        p.setPen(QtGui.QColor(20, 20, 20))
        p.drawText(x0, y + fm.ascent(), self._title)
        y += fm.height() + 4
        band_w, band_h = 18, 14
        # 顶 = max, 底 = min; 色带颜色取各分箱中心
        for i in range(self._levels):
            frac = 1.0 - (i + 0.5) / self._levels
            if self._cmap_obj is not None:
                rgba = self._cmap_obj(frac)
            else:
                rgba = (0.5, 0.5, 0.5, 1.0)
            p.fillRect(x0, y + i * band_h, band_w, band_h,
                       QtGui.QColor.fromRgbF(float(rgba[0]), float(rgba[1]),
                                             float(rgba[2])))
        f.setBold(False)
        p.setFont(f)
        for i in range(self._levels):
            val = (self._vmax - (i + 0.5) / self._levels
                   * (self._vmax - self._vmin))
            p.drawText(x0 + band_w + 6, y + i * band_h + band_h - 3,
                       "%+.3e" % val)
        p.end()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_pos = ev.position()

    def mouseMoveEvent(self, ev):
        if self._drag_pos is not None and \
                (ev.buttons() & QtCore.Qt.LeftButton):
            self.move(self.mapToParent(ev.position() - self._drag_pos))

    def mouseReleaseEvent(self, _ev):
        self._drag_pos = None


def _cmap_obj(name):
    """色图名 -> matplotlib Colormap 对象(供图例部件取色)"""
    from matplotlib import colormaps
    got = _cmap(name)
    if isinstance(got, str):
        return colormaps[got]
    return got


class ConvertDialog(QtWidgets.QDialog):
    """ODB 转换进度对话框(样式同 tmps_viewer 的导出进度框)"""

    canceled = QtCore.Signal()

    def __init__(self, parent=None, title="ODB -> FRD 转换中"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setMinimumWidth(620)
        lay = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel("准备转换 ...")
        lay.addWidget(self.label)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)              # 忙碌指示(Abaqus 启动耗时)
        lay.addWidget(self.bar)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMinimumHeight(160)
        lay.addWidget(self.log)

        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_cancel.setFixedWidth(80)
        self.btn_cancel.clicked.connect(self._on_cancel)
        lay.addWidget(self.btn_cancel, alignment=QtCore.Qt.AlignRight)

    def _on_cancel(self):
        self.btn_cancel.setEnabled(False)
        self.setLabelText("正在取消 ...")
        self.canceled.emit()

    def setLabelText(self, text):
        self.label.setText(text)
        self.append_log(text)

    def append_log(self, line):
        self.log.appendPlainText(line)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())


class InpSolveWorker(QtCore.QThread):
    """Abaqus 求解后台线程: 对 inp 执行 abaqus job=... """

    log = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    succeeded = QtCore.Signal(str)     # odb 路径
    failed = QtCore.Signal(str)

    def __init__(self, inp_path, parent=None):
        super().__init__(parent)
        self.inp_path = os.path.abspath(inp_path)
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        import subprocess
        job_dir = os.path.dirname(self.inp_path)
        job_name = os.path.splitext(os.path.basename(self.inp_path))[0]
        # ask_delete=OFF: 重复运行直接覆盖旧结果文件
        cmd = (f"abaqus job={job_name} input={job_name}.inp "
               f"interactive ask_delete=OFF")
        self.progress.emit("执行: " + cmd)
        try:
            proc = subprocess.Popen(
                cmd, cwd=job_dir, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
        except OSError as exc:
            self.failed.emit("无法启动 abaqus 命令: " + str(exc))
            return
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self.log.emit(line)
            if self._cancel:
                proc.kill()
                proc.wait()
                self.failed.emit("求解已被用户取消")
                return
        proc.wait()
        if proc.returncode != 0:
            self.failed.emit(
                f"Abaqus 求解失败(返回码 {proc.returncode}),"
                f" 请查看 {job_dir} 下的 .dat/.msg 文件")
            return
        odb_path = os.path.join(job_dir, job_name + ".odb")
        if not os.path.exists(odb_path):
            self.failed.emit("求解结束但未找到: " + odb_path)
            return
        self.succeeded.emit(odb_path)


class OdbConvertWorker(QtCore.QThread):
    """odb -> FRD 后台转换线程"""

    log = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    succeeded = QtCore.Signal(str)     # frd 路径
    failed = QtCore.Signal(str)

    def __init__(self, odb_path, fields, parent=None):
        super().__init__(parent)
        self.odb_path = odb_path
        self.fields = fields
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            frd_path = frd_convert.convert(
                self.odb_path, fields=self.fields,
                log=self.log.emit,
                progress=self.progress.emit,
                cancel=lambda: self._cancel)
            self.succeeded.emit(frd_path)
        except Exception as exc:  # noqa: BLE001
            if self._cancel:
                self.failed.emit("转换已被取消")
            else:
                self.failed.emit(str(exc))


class FrdGifWorker(QtCore.QThread):
    """FRD 云图旋转 GIF 生成(后台线程, 支持取消)"""

    progress = QtCore.Signal(int)
    log = QtCore.Signal(str)
    succeeded = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.p = params
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        p = self.p
        pl = None
        try:
            self.log.emit("创建离屏渲染器 ...")
            pl = pv.Plotter(off_screen=True,
                            window_size=(p["width"], p["height"]))
            frd_parser.apply_environment(pl, p["ibl"])
            if p["edl"]:
                pl.enable_eye_dome_lighting()
            pl.enable_anti_aliasing()

            fmt = p.get("fmt", "GIF")
            if fmt == "MP4":
                # H.264 真彩色, 无 256 色限制(需 imageio-ffmpeg)
                pl.open_movie(p["out"], framerate=p["fps"], quality=8)
                self.log.emit("MP4 真彩色输出(无调色板损失)")
            elif fmt == "GIF_HQ":
                try:
                    pl.open_gif(p["out"], fps=p["fps"], loop=0,
                                palettesize=256, quantizer=3,
                                subrectangles=True)
                    self.log.emit("GIF 高质量量化(libimagequant)")
                except Exception:  # noqa: BLE001
                    pl.open_gif(p["out"], fps=p["fps"], loop=0,
                                palettesize=256, quantizer=2,
                                subrectangles=True)
                    self.log.emit("GIF 高质量量化不可用, 回退标准量化;"
                                  " 可执行 pip install imagequant 获得最佳质量")
            else:
                pl.open_gif(p["out"], fps=p["fps"], loop=0,
                            palettesize=256, quantizer=2,
                            subrectangles=True)

            # 重建网格: 坐标为未变形基准, 位移场逐帧插值
            mesh = pv.PolyData(p["base_points"], p["faces"])
            mesh.point_data["frd_scalar"] = p["scalars"]
            mesh.points = p["base_points"] + p["U0"]   # 首帧变形系数 1

            kwargs = dict(cmap=_cmap(p["cmap"]),
                          show_scalar_bar=p.get("bar_show", True),
                          smooth_shading=p["smooth"],
                          pbr=True, metallic=p["metallic"],
                          roughness=p["roughness"], reset_camera=False,
                          scalar_bar_args=p["bar_args"])
            if p["style"] == "线框":
                kwargs.update(style="wireframe",
                              line_width=p["line_width"])
            pl.add_mesh(mesh, scalars="frd_scalar", **kwargs)
            wires, wire_bases = [], []
            if p["style"] == "曲面+线框":
                try:
                    wn = np.asarray(mesh.point_normals)
                    wsize = float(np.linalg.norm(
                        mesh.points.max(axis=0) - mesh.points.min(axis=0)))
                    weps = wsize * 2e-3
                except Exception:  # noqa: BLE001
                    wn, weps = None, 0.0
                for sign in (1.0, -1.0):        # 内外两侧各一份
                    w = mesh.copy()
                    if wn is not None:
                        w.points = np.asarray(w.points) + sign * weps * wn
                    wires.append(w)
                    # 基准 = 当前坐标(含首帧 1x 变形+外移) 减去 1x 位移
                    wire_bases.append(np.asarray(w.points).copy() - p["U0"])
                    pl.add_mesh(w, style="wireframe",
                                color=frd_parser.WIRE_COLORS[p["wire_color"]],
                                line_width=p["line_width"],
                                reset_camera=False)

            # 初始相机 = 主窗口当前视角(严格沿用, 不 reset)
            cam = pl.camera
            cam.SetFocalPoint(*p["cam_focal"])
            cam.SetPosition(*p["cam_pos"])
            cam.SetViewUp(*p["cam_up"])
            # 阻止首帧渲染时 Plotter 因"相机未标记"自动 reset 视角
            if hasattr(cam, "_is_set"):
                cam._is_set = True

            n = p["frames"]
            zoom_end = float(p.get("zoom_end", 1.0))
            for i in range(n):
                if self._cancel:
                    pl.close()
                    self.cancelled.emit()
                    return
                # 变形系数从 1 线性变化到 zoom_end(变形生长动画)
                k = 1.0 + (zoom_end - 1.0) * i / max(n - 1, 1)
                mesh.points = p["base_points"] + k * p["U0"]
                mesh.point_data["frd_scalar"] = p["scalars"] * k   # 云图值线性缩放
                for w, wb in zip(wires, wire_bases):
                    w.points = wb + k * p["U0"]   # 双侧线框同步变形
                if i > 0:
                    frd_parser.rotate_camera(pl, p["d_az"])
                pl.write_frame()
                self.progress.emit(int(100 * (i + 1) / n))
                self.log.emit(f"帧 {i + 1}/{n}")

            pl.close()
            self.succeeded.emit(p["out"])
        except Exception as exc:  # noqa: BLE001
            try:
                if pl is not None:
                    pl.close()
            except Exception:
                pass
            if self._cancel:
                self.cancelled.emit()
            else:
                import traceback
                self.log.emit(traceback.format_exc())
                self.failed.emit(f"{type(exc).__name__}: {exc}")


class FrdPlotter(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)

        # 数据状态
        self._data = None
        self._mesh = None
        self._disp_ids = None
        self._base_points = None
        self._mesh_warped = None
        self._U = None
        # 显示状态
        self._actor = None
        self._wires = []                    # 双侧线框 actor 列表
        self._edl_on = False
        self._ibl_on = False
        self._last_scalar = None       # 最近显示的 (label, arr)
        self._anim_scale = 1.0         # 动画期间云图值的线性缩放系数
        self._first_render = True      # 首次显示时 reset_camera 适配模型
        self._gif_worker = None
        self._gif_prog = None
        self._gif_cancelled = False
        self._odb_worker = None
        self._odb_prog = None
        self._solve_worker = None
        self._solve_prog = None
        self._legend = None              # 可拖动图例部件
        self.SETTINGS = QtCore.QSettings("TPMSTools", "FrdPlotter")
        self._default_bg = None       # 主题默认背景(在 plotter 创建后捕获)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)
        hbox.addWidget(self._build_panel(), 0)

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plotter = QtInteractor(self)
        self.plotter.enable_anti_aliasing()
        self._default_bg = tuple(self.plotter.background_color.float_rgb)
        vsplit.addWidget(self.plotter)
        self.logbox = QtWidgets.QPlainTextEdit(readOnly=True)
        self.logbox.setMaximumHeight(140)
        vsplit.addWidget(self.logbox)
        vsplit.setStretchFactor(0, 4)
        vsplit.setStretchFactor(1, 1)
        hbox.addWidget(vsplit, 1)

        self._build_menu()
        if self.combo_gfmt.currentText().startswith("MP4") \
                and not _mp4_available():
            self.combo_gfmt.setCurrentIndex(1)      # GIF(兼容)
            self._sync_gout_ext()
            self.statusBar().showMessage(
                "就绪(未检测到 imageio-ffmpeg, MP4 已回退 GIF; "
                "pip install imageio[ffmpeg] 可启用 MP4)")
        else:
            self.statusBar().showMessage("就绪: 文件 -> 打开 FRD")

    # ------------------------------------------------------------------
    # 菜单
    # ------------------------------------------------------------------
    def _build_menu(self):
        m_file = self.menuBar().addMenu("文件(&F)")
        act_open = QtGui.QAction("打开结果文件 ...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_dialog)
        m_file.addAction(act_open)

        act_open_odb = QtGui.QAction("打开 Abaqus ODB ...", self)
        act_open_odb.setShortcut("Ctrl+Shift+O")
        act_open_odb.triggered.connect(self.open_odb_dialog)
        m_file.addAction(act_open_odb)

        act_open_inp = QtGui.QAction("打开 Abaqus INP 并求解 ...", self)
        act_open_inp.setShortcut("Ctrl+I")
        act_open_inp.triggered.connect(self.open_inp_dialog)
        m_file.addAction(act_open_inp)
        m_file.addSeparator()
        act_quit = QtGui.QAction("退出", self)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_help = self.menuBar().addMenu("帮助(&H)")
        act_about = QtGui.QAction("关于 ...", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _show_about(self):
        QtWidgets.QMessageBox.about(
            self, "关于",
            "<b>TPMS 研究工具链 · FRD 结果显示器</b><br>版本 1.0<br><br>FRD 结果云图 / PBR 材质 / 变形显示 / MP4-GIF 动画导出 / 可配置图例<br><br><b>出处:</b> Gitee 开源项目 TPMS (https://gitee.com/sanreal/TPMS)<br><b>教程:</b> B 站 UP 主Sanreal (space.bilibili.com/1136187067)<br>使用本工具请引用出处 · 许可证: MIT")

    # ------------------------------------------------------------------
    # 控制面板
    # ------------------------------------------------------------------
    def _build_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(6)

        # ---- 结果选项树 ----
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabel("云图选项")
        self.tree.setMinimumHeight(220)
        self.tree.itemClicked.connect(self._on_tree_click)
        layout.addWidget(self.tree)

        # ---- Tab 分页: 云图 / 材质 / 变形 ----
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        def _page(title):
            w = QtWidgets.QWidget()
            fl = QtWidgets.QFormLayout(w)
            tabs.addTab(w, title)
            return fl

        # Tab 1: 云图
        f1 = _page("云图")
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(["rainbow", "plasma", "viridis",
                                  "jet", "coolwarm"])
        self.combo_cmap.currentIndexChanged.connect(self._refresh_scalar)
        f1.addRow("色图", self.combo_cmap)

        self.combo_shading = QtWidgets.QComboBox()
        self.combo_shading.addItems(["连续", "离散(分色阶)"])
        self.combo_shading.setCurrentIndex(1)
        self.combo_shading.currentIndexChanged.connect(self._refresh_scalar)
        f1.addRow("云图模式", self.combo_shading)

        self.sp_levels = QtWidgets.QSpinBox(minimum=2, maximum=64, value=11)
        self.sp_levels.valueChanged.connect(self._refresh_scalar)
        f1.addRow("离散色阶数", self.sp_levels)

        self.chk_smooth = QtWidgets.QCheckBox("平滑着色")
        self.chk_smooth.setChecked(True)
        self.chk_smooth.toggled.connect(self._refresh_scalar)
        f1.addRow(self.chk_smooth)

        self.combo_style = QtWidgets.QComboBox()
        self.combo_style.addItems(["实体曲面", "线框", "曲面+线框"])
        self.combo_style.setCurrentIndex(2)      # 默认: 曲面+线框
        self.combo_style.currentIndexChanged.connect(self._refresh_scalar)
        f1.addRow("显示样式", self.combo_style)

        roww = QtWidgets.QHBoxLayout()
        roww.addWidget(QtWidgets.QLabel("线宽"))
        self.sp_lw = QtWidgets.QSpinBox(minimum=1, maximum=10, value=1)
        self.sp_lw.valueChanged.connect(self._refresh_scalar)
        roww.addWidget(self.sp_lw)
        roww.addWidget(QtWidgets.QLabel("线色"))
        self.combo_wirecolor = QtWidgets.QComboBox()
        self.combo_wirecolor.addItems(["黑色", "白色", "灰色"])
        self.combo_wirecolor.currentIndexChanged.connect(self._refresh_scalar)
        roww.addWidget(self.combo_wirecolor)
        f1.addRow(roww)

        # Tab 2: 材质(与 implicit_plotter 同源)
        f2 = _page("材质")
        self.combo_mat = QtWidgets.QComboBox()
        self.combo_mat.addItems(list(frd_parser.MATERIAL_PRESETS.keys()))
        self.combo_mat.currentIndexChanged.connect(self._apply_mat_preset)
        f2.addRow("材质预设", self.combo_mat)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("金属度"))
        self.sp_metallic = QtWidgets.QDoubleSpinBox(
            minimum=0.0, maximum=1.0, value=0.0, decimals=2, singleStep=0.05)
        self.sp_metallic.valueChanged.connect(self._refresh_scalar)
        row.addWidget(self.sp_metallic)
        row.addWidget(QtWidgets.QLabel("粗糙度"))
        self.sp_roughness = QtWidgets.QDoubleSpinBox(
            minimum=0.0, maximum=1.0, value=0.45, decimals=2, singleStep=0.05)
        self.sp_roughness.valueChanged.connect(self._refresh_scalar)
        row.addWidget(self.sp_roughness)
        f2.addRow(row)

        self.chk_ibl = QtWidgets.QCheckBox("环境反射 IBL")
        self.chk_ibl.setChecked(True)
        self.chk_ibl.setToolTip("摄影棚式环境贴图, 提供 PBR 反射")
        self.chk_ibl.toggled.connect(self._on_ibl)
        f2.addRow(self.chk_ibl)

        self.chk_edl = QtWidgets.QCheckBox("环境光遮蔽 EDL")
        self.chk_edl.setChecked(True)
        self.chk_edl.setToolTip("增强沟壑处立体感")
        self.chk_edl.toggled.connect(self._on_edl)
        f2.addRow(self.chk_edl)

        # Tab 3: 变形
        f3 = _page("变形")
        self.chk_deform = QtWidgets.QCheckBox("变形显示(按 DISP 放大)")
        self.chk_deform.setChecked(True)
        self.chk_deform.toggled.connect(self._refresh_scalar)
        f3.addRow(self.chk_deform)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("放大系数"))
        self.sp_scale = QtWidgets.QDoubleSpinBox(
            minimum=1e-4, maximum=1e6, value=1.0, decimals=3,
            singleStep=1.0)
        self.sp_scale.valueChanged.connect(self._refresh_scalar)
        row2.addWidget(self.sp_scale)
        self.chk_autoscale = QtWidgets.QCheckBox("自动")
        self.chk_autoscale.setChecked(True)
        self.chk_autoscale.toggled.connect(self._on_autoscale)
        row2.addWidget(self.chk_autoscale)
        f3.addRow(row2)

        # Tab 4: 动画
        f4 = _page("动画")
        self.sp_gframes = QtWidgets.QSpinBox(minimum=2, maximum=500, value=36)
        f4.addRow("帧数", self.sp_gframes)
        self.sp_gfps = QtWidgets.QSpinBox(minimum=1, maximum=60, value=12)
        f4.addRow("帧率 fps", self.sp_gfps)
        self.sp_gdaz = QtWidgets.QDoubleSpinBox(minimum=-180.0, maximum=180.0,
                                                value=10.0, decimals=1,
                                                singleStep=1.0)
        self.sp_gdaz.setToolTip("每帧方位角步进; 帧数 36 x 步进 10 = 整圈")
        f4.addRow("每帧方位角°", self.sp_gdaz)
        self.sp_gzoom = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=100.0,
                                                  value=1.0, decimals=2,
                                                  singleStep=0.5)
        self.sp_gzoom.setToolTip(
            "变形生长动画: 变形放大系数从 1 线性变化到此倍数;\n"
            "1.0 = 无变化; 需勾选变形显示且 FRD 含 DISP 场")
        f4.addRow("变形倍数(末帧)", self.sp_gzoom)

        wh2 = QtWidgets.QWidget()
        hl2 = QtWidgets.QHBoxLayout(wh2)
        hl2.setContentsMargins(0, 0, 0, 0)
        self.sp_gw = QtWidgets.QSpinBox(minimum=320, maximum=4096, value=800)
        self.sp_gh = QtWidgets.QSpinBox(minimum=240, maximum=4096, value=600)
        hl2.addWidget(self.sp_gw)
        hl2.addWidget(QtWidgets.QLabel("x"))
        hl2.addWidget(self.sp_gh)
        f4.addRow("尺寸(像素)", wh2)

        self.combo_gfmt = QtWidgets.QComboBox()
        self.combo_gfmt.addItems(["MP4(真彩色, 推荐)", "GIF(兼容)", "GIF 高质量"])
        self.combo_gfmt.currentIndexChanged.connect(self._on_gfmt_changed)
        f4.addRow("输出格式", self.combo_gfmt)

        self.chk_capture = QtWidgets.QCheckBox("屏幕截图模式(所见即所得)")
        self.chk_capture.setChecked(True)
        self.chk_capture.setToolTip(
            "逐帧截取当前预览画面: 颜色/材质/图例与屏幕完全一致;\n"
            "帧尺寸 = 当前窗口大小(尺寸参数失效); 不要最小化窗口")
        f4.addRow(self.chk_capture)

        self.ed_gout = QtWidgets.QLineEdit(
            os.path.abspath("frd_animation.mp4"))
        btn_gbrowse = QtWidgets.QPushButton("浏览...")
        btn_gbrowse.clicked.connect(self._browse_gif_out)
        wg = QtWidgets.QWidget()
        hg = QtWidgets.QHBoxLayout(wg)
        hg.setContentsMargins(0, 0, 0, 0)
        hg.addWidget(self.ed_gout)
        hg.addWidget(btn_gbrowse)
        f4.addRow("输出文件", wg)

        self.btn_gif = QtWidgets.QPushButton("生成动画")
        self.btn_gif.setStyleSheet("font-weight: bold;")
        self.btn_gif.clicked.connect(self.generate_gif)
        f4.addRow(self.btn_gif)

        # Tab 5: 配置(字体 / 颜色 / 图例)
        f5 = _page("配置")
        self.combo_font = QtWidgets.QComboBox()
        self.combo_font.addItems(["arial", "courier", "times"])
        self.combo_font.setToolTip("渲染字体(Arial/Courier/Times)")
        self.combo_font.currentIndexChanged.connect(self._apply_font)
        f5.addRow("字体", self.combo_font)

        self.combo_bgcfg = QtWidgets.QComboBox()
        self.combo_bgcfg.addItems(["默认(灰蓝)", "白色", "浅灰", "黑色"])
        self.combo_bgcfg.currentIndexChanged.connect(self._apply_bg)
        f5.addRow("背景", self.combo_bgcfg)

        self.combo_barpos = QtWidgets.QComboBox()
        self.combo_barpos.addItems(
            ["浮动(可拖动)", "底部横排", "右侧竖排", "隐藏"])
        self.combo_barpos.setToolTip("图例(色标条)的位置")
        self.combo_barpos.currentIndexChanged.connect(self._on_cfg_bar)
        f5.addRow("图例位置", self.combo_barpos)

        self.sp_bartsize = QtWidgets.QSpinBox(minimum=6, maximum=20, value=10)
        self.sp_bartsize.valueChanged.connect(self._on_cfg_bar)
        f5.addRow("图例标题字号", self.sp_bartsize)

        self.sp_barlsize = QtWidgets.QSpinBox(minimum=6, maximum=16, value=8)
        self.sp_barlsize.valueChanged.connect(self._on_cfg_bar)
        f5.addRow("图例标签字号", self.sp_barlsize)

        self.sp_barn = QtWidgets.QSpinBox(minimum=2, maximum=15, value=6)
        self.sp_barn.setToolTip("色标条刻度标签数量")
        self.sp_barn.valueChanged.connect(self._on_cfg_bar)
        f5.addRow("图例刻度标签数", self.sp_barn)

        rowc = QtWidgets.QHBoxLayout()
        rowc.addWidget(QtWidgets.QLabel("宽"))
        self.sp_barw = QtWidgets.QDoubleSpinBox(
            minimum=0.1, maximum=0.9, value=0.5, decimals=2, singleStep=0.05)
        self.sp_barw.valueChanged.connect(self._on_cfg_bar)
        rowc.addWidget(self.sp_barw)
        rowc.addWidget(QtWidgets.QLabel("高"))
        self.sp_barh = QtWidgets.QDoubleSpinBox(
            minimum=0.05, maximum=0.5, value=0.09, decimals=2, singleStep=0.01)
        self.sp_barh.valueChanged.connect(self._on_cfg_bar)
        rowc.addWidget(self.sp_barh)
        f5.addRow("图例宽高", rowc)

        self.btn_apply_cfg = QtWidgets.QPushButton("应用")
        self.btn_apply_cfg.setToolTip("立即应用全部配置项(修改已实时生效, "
                                     "此按钮用于确认/兜底)")
        self.btn_apply_cfg.clicked.connect(self._apply_all_cfg)
        f5.addRow(self.btn_apply_cfg)

        self._load_cfg()

        # 底部进度条
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Tab 引用: 切到"动画"页时同步变形系数
        self.tabs = tabs
        self._page_anim = tabs.widget(3)          # 动画页
        tabs.currentChanged.connect(self._on_tab_changed)
        return panel

    def _on_tab_changed(self, index):
        """进入动画页时, 把当前变形放大系数自动写入'变形倍数(末帧)'"""
        if self.tabs.widget(index) is self._page_anim:
            self.sp_gzoom.setValue(self.sp_scale.value())
            self.statusBar().showMessage(
                f"已同步变形系数: {self.sp_scale.value():.3g}")

    # ------------------------------------------------------------------
    # 配置(字体 / 背景 / 图例)
    # ------------------------------------------------------------------
    def _load_cfg(self):
        s = self.SETTINGS
        self.combo_font.setCurrentIndex(
            int(s.value("cfg/font", 0)))
        self.combo_bgcfg.setCurrentIndex(
            int(s.value("cfg/bg", 0)))
        self.combo_barpos.setCurrentIndex(
            int(s.value("cfg/barpos", 0)))
        self.sp_bartsize.setValue(int(s.value("cfg/bartsize", 10)))
        self.sp_barlsize.setValue(int(s.value("cfg/barlsize", 8)))
        self.sp_barn.setValue(int(s.value("cfg/barn", 6)))
        self.sp_barw.setValue(float(s.value("cfg/barw", 0.5)))
        self.sp_barh.setValue(float(s.value("cfg/barh", 0.09)))

    def _save_cfg(self):
        s = self.SETTINGS
        s.setValue("cfg/font", self.combo_font.currentIndex())
        s.setValue("cfg/bg", self.combo_bgcfg.currentIndex())
        s.setValue("cfg/barpos", self.combo_barpos.currentIndex())
        s.setValue("cfg/bartsize", self.sp_bartsize.value())
        s.setValue("cfg/barlsize", self.sp_barlsize.value())
        s.setValue("cfg/barn", self.sp_barn.value())
        s.setValue("cfg/barw", float(self.sp_barw.value()))
        s.setValue("cfg/barh", float(self.sp_barh.value()))
        s.sync()

    def _apply_font(self):
        pv.global_theme.font.family = self.combo_font.currentText()
        self._save_cfg()
        self._refresh_scalar()

    def _apply_bg(self):
        idx = self.combo_bgcfg.currentIndex()
        if idx == 0:
            self.plotter.set_background(self._default_bg)
        else:
            color = {1: "white", 2: "lightgray", 3: "black"}[idx]
            self.plotter.set_background(color)
        self._save_cfg()
        self._refresh_scalar()

    def _on_cfg_bar(self, *_):
        self._save_cfg()
        self._refresh_scalar()

    def _apply_all_cfg(self):
        """显式应用配置页全部设置(各控件本身已实时生效, 此处统一刷新)"""
        pv.global_theme.font.family = self.combo_font.currentText()
        self._apply_bg()
        self._save_cfg()
        self.statusBar().showMessage("配置已应用")

    def _ensure_legend(self):
        """创建(仅一次)并返回浮动图例部件, 父控件为三维视图"""
        if self._legend is None:
            self._legend = LegendWidget(self.plotter)
            self._legend.move(10, 10)
        return self._legend

    def _update_legend(self, label, arr):
        """按当前场更新浮动图例内容并显示"""
        leg = self._ensure_legend()
        finite = arr[np.isfinite(arr)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        leg.set_data(label, vmin, vmax,
                     self.sp_barn.value(),
                     _cmap_obj(self.combo_cmap.currentText()))
        leg.show()
        leg.raise_()

    def _hide_legend(self):
        if self._legend is not None:
            self._legend.hide()

    def _bar_args(self, label, color):
        """按配置页设置构造色标条参数; 返回 None 表示隐藏"""
        pos = self.combo_barpos.currentText()
        if pos == "隐藏":
            return None
        title = label if len(label) <= 16 else label[:15] + "..."
        args = dict(title=title, color=color,
                    n_labels=self.sp_barn.value(),
                    fmt="%+.3e",                 # Abaqus 风格: +1.241e+02
                    title_font_size=self.sp_bartsize.value(),
                    label_font_size=self.sp_barlsize.value())
        if pos in ("底部横排",):
            # 横条的标签/标题画在条的上方, 需整体上移留出标签带,
            # 否则文字压到模型上互相重叠
            args.update(vertical=False,
                        width=self.sp_barw.value(),
                        height=self.sp_barh.value(),
                        position_x=max(0.0, 0.5 - self.sp_barw.value() / 2),
                        position_y=0.14)
        else:
            # 竖条: 科学计数标签较宽, 默认加宽条体防止截断
            args.update(vertical=True,
                        width=max(self.sp_barw.value(), 0.06),
                        height=max(self.sp_barh.value(), 0.72),
                        position_x=0.84, position_y=0.06)
        return args

    def _sync_gout_ext(self):
        cur = self.ed_gout.text()
        if not cur:
            return
        base = os.path.splitext(cur)[0]
        if self.combo_gfmt.currentText().startswith("MP4"):
            self.ed_gout.setText(base + ".mp4")
        else:
            self.ed_gout.setText(base + ".gif")

    def _on_gfmt_changed(self):
        """输出格式切换: MP4 无插件时警告并自动回退 GIF"""
        if self.combo_gfmt.currentText().startswith("MP4") \
                and not _mp4_available():
            self.combo_gfmt.blockSignals(True)
            self.combo_gfmt.setCurrentIndex(1)      # 回退到 GIF(兼容)
            self.combo_gfmt.blockSignals(False)
            QtWidgets.QMessageBox.warning(
                self, "缺少 MP4 后端",
                "MP4 输出需要 imageio 的 FFMPEG 插件, 当前未安装。\n\n"
                "安装命令:\n  pip install imageio[ffmpeg]\n\n"
                "已自动切换为 GIF 输出。")
        self._sync_gout_ext()

    def _browse_gif_out(self):
        is_mp4 = self.combo_gfmt.currentText().startswith("MP4")
        filt = "MP4 (*.mp4)" if is_mp4 else "GIF (*.gif)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "选择动画输出", self.ed_gout.text(), filt)
        if path:
            want = ".mp4" if is_mp4 else ".gif"
            if not path.lower().endswith(want):
                path = os.path.splitext(path)[0] + want
            self.ed_gout.setText(path)

    # ------------------------------------------------------------------
    # 槽函数
    # ------------------------------------------------------------------
    def _wire_meshes(self, show_mesh):
        """线框副本: 沿法向正负两个方向各外移一份, 内外两侧都能看见线框。

        - 体网格先提取表面(否则内部单元边会被画在曲面上, 呈虚线);
        - 法向取不到时 eps=0 兜底;
        - 网格法向朝向不统一, 单向偏移会导致部分区域线框跑到内侧,
          故正负两个方向各放一份。
        """
        try:
            if not isinstance(show_mesh, pv.PolyData):
                show_mesh = show_mesh.extract_surface()
            n = np.asarray(show_mesh.point_normals)
            size = float(np.linalg.norm(
                show_mesh.points.max(axis=0) - show_mesh.points.min(axis=0)))
            eps = size * 2e-3
        except Exception:  # noqa: BLE001
            n, eps = None, 0.0
        outs = []
        for sign in (1.0, -1.0):
            w = show_mesh.copy()
            if n is not None:
                w.points = np.asarray(w.points) + sign * eps * n
            outs.append(w)
        return outs

    def _apply_mat_preset(self):
        vals = frd_parser.MATERIAL_PRESETS.get(self.combo_mat.currentText())
        if vals is None:
            return
        self.sp_metallic.setValue(vals[0])
        self.sp_roughness.setValue(vals[1])

    def _on_ibl(self, checked):
        if checked and not self._ibl_on:
            frd_parser.apply_environment(self.plotter, True)
            self._ibl_on = True
            self._refresh_scalar()

    def _on_edl(self, checked):
        if checked and not self._edl_on:
            self.plotter.enable_eye_dome_lighting()
            self._edl_on = True

    def _on_autoscale(self, checked):
        self.sp_scale.setEnabled(not checked)
        if checked:
            self.sp_scale.setValue(self._auto_scale())

    def _auto_scale(self):
        if self._U is None or self._base_points is None:
            return 1.0
        umax = float(np.sqrt((self._U ** 2).sum(axis=1)).max())
        if umax <= 0:
            return 1.0
        size = float(np.linalg.norm(
            self._base_points.max(axis=0) - self._base_points.min(axis=0)))
        return round(0.1 * size / umax, 3)

    # ------------------------------------------------------------------
    # 文件加载与选项树
    # ------------------------------------------------------------------
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开文件", "",
            "结果/输入文件 (*.frd *.odb *.inp);;FRD (*.frd);;"
            "Abaqus ODB (*.odb);;Abaqus INP (*.inp);;所有文件 (*)")
        if not path:
            return
        low = path.lower()
        if low.endswith(".odb"):
            self.open_odb(path)
        elif low.endswith(".inp"):
            self.open_inp(path)
        else:
            self.load_frd(path)

    def open_inp_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开 Abaqus INP", "", "Abaqus INP (*.inp)")
        if path:
            self.open_inp(path)

    def open_inp(self, inp_path):
        """打开 inp: 先 Abaqus 求解 -> 得到 odb -> 转换 -> 导入"""
        if self._solve_worker is not None and self._solve_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "请稍候",
                                              "正在求解中 ...")
            return
        self._solve_worker = InpSolveWorker(inp_path, self)

        dlg = ConvertDialog(self, title="Abaqus 求解中")
        dlg.setLabelText("启动 Abaqus 求解(可能需要几分钟, 请稍候) ...")
        dlg.canceled.connect(self._solve_cancel_clicked)
        self._solve_prog = dlg

        def _log(line):
            self.logbox.appendPlainText(line)
            dlg.append_log(line)

        self._solve_worker.log.connect(_log)
        self._solve_worker.progress.connect(dlg.setLabelText)
        self._solve_worker.succeeded.connect(self._on_solve_ok)
        self._solve_worker.failed.connect(self._on_solve_fail)
        self._solve_worker.start()
        dlg.show()

    def _solve_cancel_clicked(self):
        if self._solve_worker is not None:
            self._solve_worker.request_cancel()
        if self._solve_prog is not None:
            self._solve_prog.setLabelText("正在取消 ...")

    def _on_solve_ok(self, odb_path):
        if self._solve_prog is not None:
            self._solve_prog.close()
            self._solve_prog = None
        self.logbox.appendPlainText(f"求解完成: {odb_path}, 开始转换 ...")
        self.statusBar().showMessage("求解完成, 转换中 ...")
        # 复用 odb 流程: 选场 -> 转换 -> 导入
        self.open_odb(odb_path)

    def _on_solve_fail(self, msg):
        if self._solve_prog is not None:
            self._solve_prog.close()
            self._solve_prog = None
        self.statusBar().showMessage("求解失败")
        self.logbox.appendPlainText(f"求解失败: {msg}")
        QtWidgets.QMessageBox.critical(
            self, "求解失败",
            f"{msg}\n\n请确认 abaqus 命令可用(需 Abaqus 环境)。")

    def open_odb_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开 Abaqus ODB", "", "Abaqus ODB (*.odb)")
        if path:
            self.open_odb(path)

    def open_odb(self, odb_path):
        """打开 Abaqus odb: 先转 FRD 再加载; 转换失败则提示"""
        if self._odb_worker is not None and self._odb_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "请稍候",
                                              "正在转换中 ...")
            return

        # 1) 选择转换场(默认 U, S)
        fdlg = FrdOptionsDialog(self)
        if fdlg.exec() != QtWidgets.QDialog.Accepted:
            return
        fields = fdlg.selected()
        if not fields:
            QtWidgets.QMessageBox.warning(self, "未选择场",
                                          "请至少选择一个要转换的场。")
            return

        # 2) 后台线程执行转换(不卡界面, 可取消)
        self._odb_worker = OdbConvertWorker(odb_path, fields, self)

        dlg = ConvertDialog(self)
        dlg.setLabelText(
            "启动 Abaqus 转换(Abaqus 内核启动需 10~30 秒, 请稍候) ...")
        dlg.canceled.connect(self._odb_cancel_clicked)
        self._odb_prog = dlg

        def _log(line):
            self.logbox.appendPlainText(line)
            dlg.append_log(line)

        self._odb_worker.log.connect(_log)
        self._odb_worker.progress.connect(dlg.setLabelText)
        self._odb_worker.succeeded.connect(self._on_odb_ok)
        self._odb_worker.failed.connect(self._on_odb_fail)
        self._odb_worker.start()
        dlg.show()

    def _odb_cancel_clicked(self):
        if self._odb_worker is not None:
            self._odb_worker.request_cancel()
        if self._odb_prog is not None:
            self._odb_prog.setLabelText("正在取消 ...")

    def _close_odb_prog(self):
        if self._odb_prog is not None:
            self._odb_prog.close()
            self._odb_prog = None

    def _on_odb_ok(self, frd_path):
        self._close_odb_prog()
        self.logbox.appendPlainText(f"转换完成, 加载: {frd_path}")
        self.statusBar().showMessage(f"转换完成: {frd_path}")
        self.load_frd(frd_path)

    def _on_odb_fail(self, msg):
        self._close_odb_prog()
        self.statusBar().showMessage("ODB 转换失败")
        self.logbox.appendPlainText(f"转换失败: {msg}")
        QtWidgets.QMessageBox.critical(
            self, "转换失败",
            f"无法转换:\n{msg}\n\n"
            "请确认 abaqus 命令可用(需 Abaqus 环境, 且已在 PATH 中)。")

    def load_frd(self, path):
        try:
            data = frd_parser.parse_frd(path)
            mesh, disp_ids = frd_parser.build_mesh(data)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "打开失败", str(exc))
            return

        self._data = data
        self._mesh = mesh
        self._disp_ids = disp_ids
        self._base_points = mesh.points.copy()
        self._mesh_warped = mesh.copy()
        self._U = self._latest_disp()
        self.sp_scale.setValue(self._auto_scale())
        self.setWindowTitle(f"{APP_TITLE} — {os.path.basename(path)}")

        self.plotter.clear()
        self.plotter.reset_camera()
        self._first_render = True        # 新文件首帧需重新取景(否则沿用旧视角)
        if self.chk_ibl.isChecked() and not self._ibl_on:
            frd_parser.apply_environment(self.plotter, True)
            self._ibl_on = True
        if self.chk_edl.isChecked() and not self._edl_on:
            self.plotter.enable_eye_dome_lighting()
            self._edl_on = True

        self._build_tree()
        self.logbox.appendPlainText(
            f"已加载: {path} | 节点 {mesh.n_points:,} | 单元 {mesh.n_cells:,} | "
            f"结果场 {len(data.datasets)} 个")
        self.statusBar().showMessage("点选左侧条目切换云图")

        root = self.tree.topLevelItem(1)
        first = root.child(0) if root is not None and root.childCount() else None
        if first is not None:
            while first.childCount():
                first = first.child(0)
            self.tree.setCurrentItem(first)
            self._refresh_scalar()
        else:
            # 无结果场时默认显示距离云图
            geo = self.tree.topLevelItem(0)
            if geo is not None and geo.childCount():
                it = geo.child(0)
                self.tree.setCurrentItem(it)
                self._refresh_scalar()

    def _latest_disp(self):
        if self._data is None:
            return None
        for ds in reversed(self._data.datasets):
            if ds["name"] == "DISP" and ds["ncomp"] >= 3:
                id2row = {int(nid): r for r, nid in enumerate(self._disp_ids)}
                U = np.zeros((len(self._disp_ids), 3))
                for nid, v in ds["values"].items():
                    r = id2row.get(int(nid))
                    if r is not None and len(v) >= 3:
                        U[r] = v[:3]
                return U
        return None

    def _build_tree(self):
        self.tree.clear()
        root_geo = QtWidgets.QTreeWidgetItem(self.tree, ["Geometry"])
        root_geo.setIcon(0, _icon("geom"))
        for label, key in [("到原点距离", "dist"), ("X 坐标", "x"),
                           ("Y 坐标", "y"), ("Z 坐标", "z")]:
            it = QtWidgets.QTreeWidgetItem(root_geo, [label])
            it.setIcon(0, _icon("leaf"))
            it.setData(0, QtCore.Qt.UserRole, ("geom", key))
        root_geo.setExpanded(True)

        root_res = QtWidgets.QTreeWidgetItem(self.tree, ["Field outputs"])
        root_res.setIcon(0, _icon("root"))
        groups = {}
        for idx, ds in enumerate(self._data.datasets):
            groups.setdefault(ds["name"], []).append(idx)
        for name, idxs in groups.items():
            fnode = QtWidgets.QTreeWidgetItem(root_res, [name])
            fnode.setIcon(0, _icon(name))            # 按场类型分色/形状
            multi = len(idxs) > 1
            for idx in idxs:
                ds = self._data.datasets[idx]
                parent = fnode
                if multi:
                    parent = QtWidgets.QTreeWidgetItem(
                        fnode, [f"t = {ds['time']:.4g}"])
                    parent.setIcon(0, _icon("time"))
                for comp in frd_parser.components_for(ds):
                    it = QtWidgets.QTreeWidgetItem(parent, [comp])
                    it.setIcon(0, _icon("leaf"))
                    it.setData(0, QtCore.Qt.UserRole, (idx, comp))
        root_res.setExpanded(True)
        self.tree.expandAll()

    # ------------------------------------------------------------------
    # 云图刷新
    # ------------------------------------------------------------------
    def _on_tree_click(self, item, _col):
        self.tree.setCurrentItem(item)
        self._refresh_scalar()

    def _current_scalar(self):
        item = self.tree.currentItem()
        if item is None or self._mesh is None:
            return None
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None:
            return None
        if payload[0] == "geom":
            key = payload[1]
            pts = self._mesh.points
            if key == "dist":
                return "Distance", np.linalg.norm(pts, axis=1)
            axis = {"x": 0, "y": 1, "z": 2}[key]
            return f"{key.upper()} coord", pts[:, axis]
        idx, comp = payload
        ds = self._data.datasets[idx]
        arr = frd_parser.component_array(ds, comp, self._disp_ids)
        label = f"{ds['name']}, {comp}"        # Abaqus 风格标题
        if len(self._data.datasets) > 1:
            label += f" (t={ds['time']:.4g})"
        return label, arr

    def _refresh_scalar(self):
        got = self._current_scalar()
        if got is None:
            got = self._last_scalar          # 无选中时沿用上次的场
        if got is None or self._mesh is None:
            return
        label, arr = got
        self._last_scalar = got
        # 动画期间云图值随变形系数线性缩放
        arr = np.asarray(arr, dtype=float) * self._anim_scale

        # 变形
        if self.chk_deform.isChecked() and self._U is not None:
            show_mesh = self._mesh_warped
            self._mesh_warped.points = (self._base_points
                                        + self.sp_scale.value() * self._U)
        else:
            show_mesh = self._mesh
        show_mesh.point_data["frd_scalar"] = arr

        # 色标文字颜色随背景自适应
        bg = np.asarray(self.plotter.background_color.float_rgb)
        bar_color = "white" if bg.mean() < 0.5 else "black"

        style = self.combo_style.currentText()

        # 浮动图例: 用可拖动 Qt 部件替代 VTK 色标条
        floating = self.combo_barpos.currentText() == "浮动(可拖动)"
        if floating:
            bar_args = None
            self._update_legend(label, arr)
        else:
            self._hide_legend()
            bar_args = self._bar_args(label, bar_color)

        kwargs = dict(
            cmap=_cmap(self.combo_cmap.currentText()),
            show_scalar_bar=bar_args is not None,
            smooth_shading=self.chk_smooth.isChecked(),
            reset_camera=False,
            pbr=True,
            metallic=self.sp_metallic.value(),
            roughness=self.sp_roughness.value(),
        )
        if bar_args is not None:
            kwargs.update(scalar_bar_args=bar_args)
        if style == "线框":
            kwargs.update(style="wireframe", line_width=self.sp_lw.value())
        if self.combo_shading.currentText().startswith("离散") \
                and style != "线框":
            kwargs.update(n_colors=self.sp_levels.value(),
                          clim=[float(np.nanmin(arr)),
                                float(np.nanmax(arr))])

        # 删除所有旧图例 actor(与 PyVista 版本无关, 扫描渲染器逐项移除)
        for act in list(self.plotter.actors.values()):
            if act is None:
                continue
            cls = act.GetClassName() or ""
            if "ScalarBar" in cls:
                self.plotter.remove_actor(act, reset_camera=False)
        # 关键: 同时清空 PyVista 的图例登记簿;
        # 否则同名标题会被认为"已存在"而跳过创建, 导致时有时无
        try:
            self.plotter.scalar_bars.clear()
        except Exception:  # noqa: BLE001
            try:      # 兜底: 逐个删除登记项
                for k in list(self.plotter.scalar_bars.keys()):
                    self.plotter.scalar_bars.pop(k, None)
            except Exception:  # noqa: BLE001
                pass

        for a in [self._actor] + self._wires:
            if a is not None:
                self.plotter.remove_actor(a, reset_camera=False)
        self._actor = self.plotter.add_mesh(show_mesh,
                                            scalars="frd_scalar",
                                            reset_camera=self._first_render,
                                            **{k: v for k, v in kwargs.items()
                                               if k != "reset_camera"})
        self._first_render = False
        self._wires = []
        if style == "曲面+线框":
            for w in self._wire_meshes(show_mesh):
                self._wires.append(self.plotter.add_mesh(
                    w, style="wireframe",
                    color=frd_parser.WIRE_COLORS[self.combo_wirecolor.currentText()],
                    line_width=self.sp_lw.value(), reset_camera=False))
        self.plotter.render()

        finite = arr[np.isfinite(arr)]
        if finite.size:
            self.statusBar().showMessage(
                f"{label} | min {finite.min():.4g} | max {finite.max():.4g}")

    # ------------------------------------------------------------------
    # GIF 动画导出
    # ------------------------------------------------------------------
    def _capture_animation(self, p):
        """屏幕截图模式: 逐帧截取当前预览窗口, 编码为 GIF/MP4。

        所见即所得: 颜色/材质/IBL/图例与屏幕完全一致, 不受离屏
        渲染与 256 色调色板差异影响(GIF 格式本身的量化仍会有,
        建议 MP4)。
        """
        try:
            import imageio
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "缺少 imageio", "截图模式需要 imageio 库")
            return

        out, fmt = p["out"], p["fmt"]
        try:
            if fmt == "MP4":
                writer = imageio.get_writer(out, fps=p["fps"], quality=8)
            elif fmt == "GIF_HQ":
                try:
                    writer = imageio.get_writer(
                        out, fps=p["fps"], loop=0, palettesize=256,
                        quantizer=3, subrectangles=True)
                except Exception:  # noqa: BLE001
                    writer = imageio.get_writer(
                        out, fps=p["fps"], loop=0, palettesize=256,
                        quantizer=2, subrectangles=True)
            else:
                writer = imageio.get_writer(
                    out, fps=p["fps"], loop=0, palettesize=256,
                    quantizer=2, subrectangles=True)
        except Exception as exc:  # noqa: BLE001
            hint = ("\n\nMP4 输出需要: pip install imageio[ffmpeg]"
                    if fmt == "MP4" else "")
            QtWidgets.QMessageBox.critical(
                self, "创建输出失败", f"无法创建 {out}:\n{exc}{hint}")
            return

        prog = QtWidgets.QProgressDialog("截图模式生成中 ...", "取消",
                                         0, p["frames"], self)
        prog.setWindowTitle("动画生成中")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.show()

        n = p["frames"]
        cancelled = False
        U = self._U if (self.chk_deform.isChecked()
                        and self._U is not None) else None
        zoom_end = float(p["zoom_end"])
        orig_scale = self.sp_scale.value()      # 结束后恢复
        ok = True
        try:
            for i in range(n):
                if prog.wasCanceled():
                    cancelled = True
                    break
                # 变形生长: 逐帧把变形系数写入界面控件(阻塞信号防递归),
                # _refresh_scalar 会按该系数重建变形网格
                if U is not None:
                    k = 1.0 + (zoom_end - 1.0) * i / max(n - 1, 1)
                    self.sp_scale.blockSignals(True)
                    self.sp_scale.setValue(k)
                    self.sp_scale.blockSignals(False)
                    self._anim_scale = k          # 云图值同步线性缩放
                if i > 0:
                    frd_parser.rotate_camera(self.plotter, p["d_az"])
                self._refresh_scalar()
                QtWidgets.QApplication.processEvents()

                pm = self.plotter.grab()
                writer.append_data(_pixmap_to_np(pm))
                prog.setValue(i + 1)
                QtWidgets.QApplication.processEvents()
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
            # 恢复现场: 转回初始视角与变形系数
            if U is not None:
                self.sp_scale.blockSignals(True)
                self.sp_scale.setValue(orig_scale)
                self.sp_scale.blockSignals(False)
            self._anim_scale = 1.0
            frd_parser.rotate_camera(self.plotter, -p["d_az"] * (n - 1))
            self._refresh_scalar()

        prog.close()
        if cancelled:
            self.statusBar().showMessage("截图生成已取消(文件可能不完整)")
            return
        if ok:
            self.logbox.appendPlainText(f"动画完成(截图模式): {out}")
            self.statusBar().showMessage(f"动画已生成: {out}")
            QtWidgets.QMessageBox.information(
                self, "生成完成", f"动画已保存:\n{out}")
        else:
            self.logbox.appendPlainText(f"截图生成失败: {err}")
            QtWidgets.QMessageBox.critical(self, "生成失败", err)

    def generate_gif(self):
        if self._gif_worker is not None and self._gif_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "请稍候", "正在生成中 ...")
            return
        got = self._current_scalar()
        if got is None or self._mesh is None:
            QtWidgets.QMessageBox.warning(
                self, "无法生成", "请先打开 FRD 并在左侧选择一个云图条目。")
            return
        label, arr = got

        # 当前显示的网格(含变形)与视角; 体网格取表面
        if self.chk_deform.isChecked() and self._U is not None:
            show_mesh = self._mesh_warped
            self._mesh_warped.points = (self._base_points
                                        + self.sp_scale.value() * self._U)
        else:
            show_mesh = self._mesh
        if not isinstance(show_mesh, pv.PolyData):
            show_mesh = show_mesh.extract_surface()

        gfmt = {"MP4(真彩色, 推荐)": "MP4",
                "GIF(兼容)": "GIF",
                "GIF 高质量": "GIF_HQ"}[self.combo_gfmt.currentText()]
        if gfmt == "MP4" and not _mp4_available():
            QtWidgets.QMessageBox.warning(
                self, "缺少 MP4 后端",
                "MP4 输出需要 imageio 的 FFMPEG 插件, 当前未安装。\n\n"
                "安装命令:\n  pip install imageio[ffmpeg]\n\n"
                "或改用 GIF 输出。")
            return
        out = self.ed_gout.text()
        want_ext = ".mp4" if gfmt == "MP4" else ".gif"
        if not out.lower().endswith(want_ext):
            out = os.path.splitext(out)[0] + want_ext
            self.ed_gout.setText(out)
        out_dir = os.path.dirname(os.path.abspath(out))
        if not os.path.isdir(out_dir):
            QtWidgets.QMessageBox.warning(self, "路径错误",
                                          f"目录不存在:\n{out_dir}")
            return

        bg = np.asarray(self.plotter.background_color.float_rgb)
        bar_color = "white" if bg.mean() < 0.5 else "black"
        bar_args = self._bar_args(label, bar_color)

        U0 = self._U if (self.chk_deform.isChecked()
                         and self._U is not None) else np.zeros_like(
                             self._base_points)
        params = {
            "base_points": np.asarray(self._base_points).copy(),
            "faces": np.asarray(show_mesh.faces).copy(),
            "U0": np.asarray(U0).copy(),
            "scalars": np.asarray(arr).copy(),
            "label": label,
            "cmap": self.combo_cmap.currentText(),
            "smooth": self.chk_smooth.isChecked(),
            "style": self.combo_style.currentText(),
            "bar_show": bar_args is not None,
            "bar_args": dict(bar_args) if bar_args else {},
            "line_width": self.sp_lw.value(),
            "wire_color": self.combo_wirecolor.currentText(),
            "metallic": self.sp_metallic.value(),
            "roughness": self.sp_roughness.value(),
            "ibl": self.chk_ibl.isChecked(),
            "edl": self.chk_edl.isChecked(),
            "frames": self.sp_gframes.value(),
            "fps": self.sp_gfps.value(),
            "d_az": self.sp_gdaz.value(),
            "zoom_end": self.sp_gzoom.value(),
            "fmt": gfmt,
            "width": self.sp_gw.value(),
            "height": self.sp_gh.value(),
            "out": out,
            "cam_pos": list(self.plotter.camera.position),
            "cam_focal": list(self.plotter.camera.focal_point),
            "cam_up": list(frd_parser._viewup(self.plotter.camera)),
        }

        params["capture"] = self.chk_capture.isChecked()

        if params["capture"]:
            # 屏幕截图模式: 同步执行, 逐帧截取当前窗口
            self._capture_animation(params)
            return

        self._gif_cancelled = False
        self.btn_gif.setEnabled(False)
        self._gif_worker = FrdGifWorker(params, self)

        prog = QtWidgets.QProgressDialog("准备 GIF 生成 ...", "取消",
                                         0, 100, self)
        prog.setWindowTitle("GIF 生成中")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.canceled.connect(self._gif_cancel_clicked)
        self._gif_prog = prog

        self._gif_worker.progress.connect(self.progress.setValue)
        self._gif_worker.progress.connect(prog.setValue)
        self._gif_worker.log.connect(self.logbox.appendPlainText)
        self._gif_worker.succeeded.connect(self._on_gif_ok)
        self._gif_worker.failed.connect(self._on_gif_fail)
        self._gif_worker.cancelled.connect(self._on_gif_cancelled)
        self._gif_worker.start()
        prog.show()

    def _gif_cancel_clicked(self):
        self._gif_cancelled = True
        if self._gif_worker is not None:
            self._gif_worker.request_cancel()
        if self._gif_prog is not None:
            self._gif_prog.setLabelText("正在取消 ...")

    def _close_gif_prog(self):
        if self._gif_prog is not None:
            self._gif_prog.close()
            self._gif_prog = None

    def _on_gif_ok(self, path):
        self.btn_gif.setEnabled(True)
        self._close_gif_prog()
        self.logbox.appendPlainText(f"GIF 完成: {path}")
        self.statusBar().showMessage(f"GIF 已生成: {path}")
        QtWidgets.QMessageBox.information(
            self, "生成完成", f"GIF 已保存:\n{path}")

    def _on_gif_fail(self, msg):
        self.btn_gif.setEnabled(True)
        self._close_gif_prog()
        if self._gif_cancelled:
            return
        self.logbox.appendPlainText(f"GIF 失败: {msg}")
        QtWidgets.QMessageBox.critical(self, "生成失败", msg)

    def _on_gif_cancelled(self):
        self.btn_gif.setEnabled(True)
        self._close_gif_prog()
        self.statusBar().showMessage("GIF 生成已取消")
        QtWidgets.QMessageBox.information(
            self, "已取消", "GIF 生成已取消。")


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()

    app = QtWidgets.QApplication(sys.argv)
    win = FrdPlotter()
    win.show()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        win.load_frd(sys.argv[1])
    sys.exit(app.exec())
