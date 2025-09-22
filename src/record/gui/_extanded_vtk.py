import numpy as np
import vtk
from PyQt6 import QtWidgets, QtGui, QtCore

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderer,
    vtkActor,
    vtkRenderWindowInteractor,
    vtkPolyDataMapper,
)
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLPolyDataMapper

# from vtkmodules.vtkProperty import vtkProperty
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkCommonDataModel import vtkPolyData


class SceneVTKQT:
    def __init__(
        self,
        frame_holder,
        vtk_widget_name="vtkwidget_name",
        source_plane_size=10,
        *args,
        **kwargs,
    ):
        self.frame_holder = frame_holder
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame_holder)
        self.vl_stl = QtWidgets.QVBoxLayout(self.frame_holder)
        self.vl_stl.setObjectName(vtk_widget_name)
        self.vl_stl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.vl_stl.setStretch(0, 2)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        self.frame_holder.setLayout(self.vl_stl)

        self.actors_pool = {}
        self.scale = 1.0
        self.source_plane_size = source_plane_size

    def set_button_press_event(self, event_name, func):
        self.iren.AddObserver(event_name, func)

    def create_plane_source(self):
        # Définir les couleurs
        colors = vtkNamedColors()

        # Créer un plan qui sert de sol
        plane_source = vtkPlaneSource()
        plane_source.SetXResolution(self.source_plane_size)
        plane_source.SetYResolution(self.source_plane_size)
        plane_source.SetOrigin(
            -self.source_plane_size / 2, -self.source_plane_size / 2, 0
        )  # Déplacer pour centrer
        plane_source.SetPoint1(
            self.source_plane_size / 2, -self.source_plane_size / 2, 0
        )
        plane_source.SetPoint2(
            -self.source_plane_size / 2, self.source_plane_size / 2, 0
        )
        plane_source.Update()

        # Mapper et activer le quadrillage
        plane_mapper = vtkPolyDataMapper()
        plane_mapper.SetInputConnection(plane_source.GetOutputPort())

        # Créer l'acteur pour le sol
        plane_actor = vtkActor()
        plane_actor.SetMapper(plane_mapper)
        plane_actor.GetProperty().SetColor(colors.GetColor3d("LightGray"))
        plane_actor.GetProperty().SetOpacity(0.5)  # Transparence
        plane_actor.GetProperty().SetEdgeVisibility(
            True
        )  # Activer les lignes pour le quadrillage
        plane_actor.GetProperty().SetEdgeColor(colors.GetColor3d("Black"))

        # Ajouter l'acteur à la scène
        self.ren.AddActor(plane_actor)
        self.ren.SetBackground(colors.GetColor3d("SlateGray"))

    def add_axes_actor(
        self,
        actor_name,
        tvec=np.array([0, 0, 0]),
        XAxisLabelText="",
        YAxisLabelText="",
        ZAxisLabelText="",
    ):
        axes_transfom = vtkTransform()
        axes_transfom.Translate(*tvec)
        axes_actor = vtkAxesActor()
        axes_actor.SetXAxisLabelText(XAxisLabelText)
        axes_actor.SetYAxisLabelText(YAxisLabelText)
        axes_actor.SetZAxisLabelText(ZAxisLabelText)
        axes_actor.SetUserTransform(axes_transfom)
        self.add_actor(actor_name=f"{actor_name}", actor_obj=axes_actor)

    def add_actor(self, actor_name: str, actor_obj):
        if type(actor_obj) != list:
            actor_obj = [actor_obj]

        self.actors_pool.update({actor_name: actor_obj})

        for actor in actor_obj:
            self.ren.AddActor(actor)

    def remove_actor(self, actor_name):
        actor_list = self.actors_pool[actor_name]
        for actor in actor_list:
            self.ren.RemoveActor(actor)
        del self.actors_pool[actor_name]

    def get_actor(self, actor_name):
        return self.actors_pool[actor_name][0]

    def set_camera_orientation_postion(
        self,
        view_up=np.array([0, 0, 1]),
        focal_point=np.array([0, 0, 0]),
        position=np.array([10, 0.5, 3]),
    ):
        camera = vtk.vtkCamera()
        camera.SetViewUp(view_up)
        camera.SetFocalPoint(focal_point)
        camera.SetPosition(position)
        self.ren.SetActiveCamera(camera)

    @staticmethod
    def set_actor_orientation_position(
        actor, rvec, tvec, default_orientation=np.zeros(3), scale=1.0
    ):
        def _set_actor_orientation_postion(
            actor, rvec, tvec, default_orientation, scale
        ):
            actor.SetPosition(
                tvec[0] * (1 / scale),
                tvec[1] * (1 / scale),
                tvec[2] * (1 / scale),
            )

            W = np.degrees(np.linalg.norm(rvec))
            X, Y, Z = np.zeros(3)
            if W != 0:
                X = rvec[0] / np.linalg.norm(rvec)
                Y = rvec[1] / np.linalg.norm(rvec)
                Z = rvec[2] / np.linalg.norm(rvec)
            actor.SetOrientation(default_orientation)
            actor.RotateWXYZ(W, X, Y, Z)

        if type(actor) == list:
            for actor in actor:
                _set_actor_orientation_postion(
                    actor, rvec, tvec, default_orientation, scale
                )
        else:
            _set_actor_orientation_postion(
                actor, rvec, tvec, default_orientation, scale
            )

    @staticmethod
    def get_actor_orientation_position(actor, rodrigues=False):
        vtkTransform = actor.GetUserTransform()
        if rodrigues:
            tvec = vtkTransform.GetPosition()
            a, b, c, d = actor.GetOrientationWXYZ()
            theta = np.radians(a)
            rvec = np.array([b, c, d]) * theta
            return rvec.reshape(3), np.array(tvec).reshape(3)
        else:
            return actor.GetOrientationWXYZ(), vtkTransform.GetPosition()


def create_Stl_Actor(
    filename="./",
    color=(0 / 255, 255 / 255, 0 / 255),
    alpha=1,
    *args,
    **kwargs,
):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(alpha)
    return actor


def create_Points(
    p3d=np.zeros((2, 3)),
    color=(255 / 255, 255 / 255, 255 / 255),
    alpha=1,
    point_size=1,
    *args,
    **kwargs,
):
    p3d = p3d.reshape(-1, 3)
    if p3d.shape[0] == 1:
        p3d = np.concatenate((p3d, p3d), axis=0)

    # Create the geometry of a point (the coordinate)
    points = vtk.vtkPoints()
    # p = [1.0, 2.0, 3.0]

    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    for p in p3d:
        id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)

    # Create a polydata object
    point = vtk.vtkPolyData()

    # Set the points and vertices we created as the geometry and topology of the polydata
    point.SetPoints(points)
    point.SetVerts(vertices)

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(point)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(alpha)
    actor.GetProperty().SetPointSize(point_size)

    return actor


def create_Line(
    origin=np.zeros(3),
    destination=np.zeros(3),
    color=(255 / 255, 255 / 255, 255 / 255),
    alpha=1,
):
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(2)
    points.SetPoint(0, origin)
    points.SetPoint(1, destination)

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(2)
    lines.InsertCellPoint(0)
    lines.InsertCellPoint(1)

    polygon = vtk.vtkPolyData()
    polygon.SetPoints(points)
    polygon.SetLines(lines)
    polygonMapper = vtk.vtkPolyDataMapper()
    polygonMapper.SetInputData(polygon)
    polygonMapper.Update()

    lineActor = vtk.vtkActor()
    lineActor.SetMapper(polygonMapper)
    lineActor.GetProperty().SetColor(color)
    lineActor.GetProperty().SetOpacity(alpha)

    return lineActor


def plot_Base(base_arrow_scale=0.05, base=None, scale=1.0e3, **kwargs):
    if base is None:
        base = np.concatenate([np.zeros(3), np.eye(3).flatten()]).reshape(-1, 3)

    origin, u, v, w = base

    u_n = (u - origin) / np.linalg.norm(u - origin)
    v_n = (v - origin) / np.linalg.norm(v - origin)
    w_n = (w - origin) / np.linalg.norm(w - origin)
    base_arrow_scale *= scale
    lineActor_X = create_Line(
        origin=origin,
        destination=origin + u_n * base_arrow_scale,
        color=(255 / 255, 0 / 255, 0 / 255),
        alpha=1,
    )
    lineActor_Y = create_Line(
        origin=origin,
        destination=origin + v_n * base_arrow_scale,
        color=(0 / 255, 255 / 255, 0 / 255),
        alpha=1,
    )
    lineActor_Z = create_Line(
        origin=origin,
        destination=origin + w_n * base_arrow_scale,
        color=(0 / 255, 0 / 255, 255 / 255),
        alpha=1,
    )

    return [lineActor_X, lineActor_Y, lineActor_Z]


def create_Sphere(
    radius=1,
    center=np.zeros(3),
    PhiResolution=100,
    ThetaResolution=100,
    color=(1, 1, 1),
    alpha=1,
    return_type="actor",
    *args,
    **kwargs,
):
    """_summary_

    Args:
        radius (int, optional): _description_. Defaults to 1.
        center (_type_, optional): _description_. Defaults to np.zeros(3).
        PhiResolution (int, optional): _description_. Defaults to 100.
        ThetaResolution (int, optional): _description_. Defaults to 100.
        color (tuple, optional): _description_. Defaults to (1, 1, 1).
        alpha (int, optional): _description_. Defaults to 1.
        return_type (str, optional): _description_. Defaults to "actor".

    Returns:
        _type_: _description_
    """
    # Create a sphere
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(center)
    sphereSource.SetRadius(radius)
    # Make the surface smooth.
    sphereSource.SetPhiResolution(PhiResolution)
    sphereSource.SetThetaResolution(ThetaResolution)
    sphereSource.Update()
    if return_type == "polyData":
        return sphereSource.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(alpha)

    if return_type == "actor":
        return actor


def create_spheres_node(
    p3d=np.zeros((2, 3)), color=(1, 1, 1), alpha=1, radius=1, *args, **kwargs
):
    appender = vtk.vtkAppendPolyData()
    reader = vtk.vtkUnstructuredGridReader()

    for p in p3d:
        polydata = create_Sphere(
            center=p, radius=radius, return_type="polyData", *args, **kwargs
        )
        input1 = vtk.vtkPolyData()
        input1.ShallowCopy(polydata)

        appender.AddInputData(input1)

    appender.Update()

    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(appender.GetOutputPort())
    cleanFilter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cleanFilter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(alpha)

    return actor
