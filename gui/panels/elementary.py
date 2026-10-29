"""Elementary control panel (grating / Fresnel / wavelength)."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import Meadowlark_SLM.parameters as param
from MyQtLineEdit.myqtLineEdit import MyQtLineEdit



class ElementaryControlPanel(QtWidgets.QWidget):
    """Owns this panel's widgets and handlers. Shared phase state is on ``self.gui``."""

    def __init__(self, gui, parent=None):
        super().__init__(parent)
        self.gui = gui
        self._build()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.root)

    def _build(self):
        """Grating / Fresnel / wavelength column (`self.w6`)."""
        self.root = pg.LayoutWidget()
        self.w6 = self.root

        self.label_w6 = QtWidgets.QLabel(""" -- ELEMENTARY CONTROL -- """)
        self.label_w6.setAlignment(QtCore.Qt.AlignCenter)
        # self.moveLabel = QtWidgets.QLabel('GRATING')
        # self.moveLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.upBtn = QtWidgets.QPushButton('UP')
        # self.downBtn = QtWidgets.QPushButton('DOWN')
        # self.leftBtn = QtWidgets.QPushButton('LEFT')
        # self.rightBtn = QtWidgets.QPushButton('RIGHT')
        # self.stepGratingEdit = MyQtLineEdit('1')
        # self.stepGratingEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.positionLabel = QtWidgets.QLabel('Grating shift (in um)')
        self.position_xEdt = MyQtLineEdit(str(self.gui.dx))
        self.position_yEdt = MyQtLineEdit(str(self.gui.dy))
        # self.fresnelLabel = QtWidgets.QLabel('FRESNEL LENS')
        # self.fresnelLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.fresnelEpsilonLabel = QtWidgets.QLabel('Defocus')
        # self.fresnelEpsilonEdt = MyQtLineEdit('0')
        self.paramPLabel = QtWidgets.QLabel('P parameter')
        self.paramPEdt = MyQtLineEdit(str(param.P))
        self.paramWavelengthLabel = QtWidgets.QLabel('Wavelength (nm)')
        self.paramWavelengthEdit = MyQtLineEdit('%.0f'%(param.lamb*1e9))

        self.w6.addWidget(self.label_w6, row=0, colspan=4)
        self.w6.addWidget(self.positionLabel, row=1, col=0)
        self.w6.addWidget(self.position_xEdt, row=1, col=1)
        self.w6.addWidget(self.position_yEdt, row=1, col=2)
        self.w6.addWidget(self.paramPLabel, row=2, col=0)
        self.w6.addWidget(self.paramPEdt, row=2, col=1)
        self.w6.addWidget(self.paramWavelengthLabel, row=2, col=2)
        self.w6.addWidget(self.paramWavelengthEdit, row=2, col=3)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        # self.upBtn.clicked.connect(self.moveUp)
        # self.downBtn.clicked.connect(self.moveDown)
        # self.leftBtn.clicked.connect(self.moveLeft)
        # self.rightBtn.clicked.connect(self.moveRight)
        # self.stepGratingEdit.returnPressed.connect(self.updateStepGrating)
        self.position_xEdt.returnPressed.connect(self.updatePositionGrating)
        self.position_yEdt.returnPressed.connect(self.updatePositionGrating)
        # self.fresnelEpsilonEdt.returnPressed.connect(self.createFresnelPattern)
        self.paramPEdt.returnPressed.connect(self.updateParamP) #2020/01/27
        self.paramWavelengthEdit.returnPressed.connect(self.updateParamWavelength)

    def moveLeft(self):
        self.gui.dx = self.gui.dx - self.gui.deltax
        self.createGratingPattern(self.gui.dx, self.gui.dy)

    def moveRight(self):
        self.gui.dx = self.gui.dx + self.gui.deltax
        self.createGratingPattern(self.gui.dx, self.gui.dy)

    def moveUp(self):
        self.gui.dy = self.gui.dy + self.gui.deltay
        self.createGratingPattern(self.gui.dx, self.gui.dy)

    def moveDown(self):
        self.gui.dy = self.gui.dy - self.gui.deltay
        self.createGratingPattern(self.gui.dx, self.gui.dy)

    @staticmethod
    def _set_float_line_edit(edit, value):
        """
        Write ``value`` into a float line edit without disturbing the text/cursor
        when the numeric value is already the same (e.g. Enter after Up/Down).
        When the value does change, keep the existing decimal precision unless
        that would round it (e.g. a remote HERO call with more decimals).
        """
        value = float(value)
        text = edit.text().strip()
        try:
            if abs(float(text) - value) <= 1e-12 * max(1.0, abs(value)):
                return
        except ValueError:
            pass
        cpos = edit.cursorPosition()
        decimalpos = text.find(".")
        if decimalpos >= 0:
            precision = max(0, len(text) - decimalpos - 1)
            newtext = f"{value:.{precision}f}"
        elif abs(value - round(value)) < 1e-12:
            newtext = str(int(round(value)))
        else:
            newtext = repr(value)
        if abs(float(newtext) - value) > 1e-12 * max(1.0, abs(value)):
            newtext = repr(value)
        edit.setText(newtext)
        edit.setCursorPosition(min(max(0, cpos + len(newtext) - len(text)), len(newtext)))

    def updateHorizontalGrating(self,x):
        self.gui.dx = x
        self.createGratingPattern(self.gui.dx, self.gui.dy)

    def updateVerticalGrating(self,y):
        self.gui.dy = y
        self.createGratingPattern(self.gui.dx, self.gui.dy)

    def updatePositionGrating(self):
        '''
        Updates the grating with the values in the interface
        '''
        self.gui.dx = float(self.position_xEdt.text())
        self.gui.dy = float(self.position_yEdt.text())
        self.createGratingPattern(self.gui.dx, self.gui.dy)

    def updateStepGrating(self):
        '''Updates the grating step when a return is pressed in the field'''
        self.gui.deltax = float(self.stepGratingEdit.text())
        self.gui.deltay = self.gui.deltax

    def updateParamP(self):
        '''Updates the P parameter when a return is pressed in the field'''
        self.gui.param_P = float(self.paramPEdt.text())
        self.gui.updatePhase()

    def updateParamWavelength(self):
        param.lamb = float(self.paramWavelengthEdit.text())*1e-9
        position_to_step_per_pixel = 2*np.pi/param.lamb*(param.M/param.f)*param.pslm
    def createGratingPattern(self, dx, dy):
        '''
        Create the grating phase pattern with user-defined wavevectors kx and ky

        (1) To shift the trap by a distance d, we need to deflect the beam by an angle
        alpha = M * d / f

        with f the effective focal length of the objective
        and M the telescope magnification between SLM and objective

        (2) A grating with a phase shift of phi between two pixels gives a deflection of
        alpha = (phi / 2pi * wavelength) / pixel separation

        This gives us the change of phase from pixel to pixel to shift the trap by a distance d:

        phi = 2pi * M/f * d * pixel separation / wavelength
        '''
        self._set_float_line_edit(self.position_xEdt, dx)
        self._set_float_line_edit(self.position_yEdt, dy)

        dphi_x = 2 * np.pi / param.lamb * param.M / param.f * dx * 1e-6
        dphi_y = 2 * np.pi / param.lamb * param.M / param.f * dy * 1e-6

        self.gui.im_grating  = self.gui.X * dphi_x + self.gui.Y * dphi_y
        self.gui.updatePhase()

    def createFresnelPattern(self, dz = None):
        '''
        Creates a Fresnel lens pattern with a user-defined defocus
        '''
        if dz == None:
            dz = float(self.fresnelEpsilonEdt.text())
        else:
            self._set_float_line_edit(self.fresnelEpsilonEdt, dz)

        self.gui.fresnelPhase = np.pi/param.lamb * (param.M/param.f)**2 * dz*1e-6 * (self.gui.X**2+self.gui.Y**2)
        self.gui.updatePhase()
