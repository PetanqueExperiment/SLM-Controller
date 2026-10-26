"""Advanced control panel (corrections, pupil, dormant TEM01/SO/LG)."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import Meadowlark_SLM.parameters as param
from MyQtLineEdit.myqtLineEdit import MyQtLineEdit
from gui.constants import PUPIL_VERTICAL_OFFSET, position_to_step_per_pixel
from gpu_module_v7_3 import trap_phase_to_slm



class AdvancedControlPanel(QtWidgets.QWidget):
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
        """Corrections, pupil, traps, SO, LG (`self.w7`)."""
        self.root = pg.LayoutWidget()
        self.Advanced_control_layout = self.root

        self.label_Advanced_control_layout = QtWidgets.QLabel(""" -- ADVANCED CONTROL -- """)
        self.label_Advanced_control_layout.setAlignment(QtCore.Qt.AlignCenter)
        self.correctionLabel = QtWidgets.QLabel('CORRECTION')
        self.correctionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ManufacturerCorrectionChkBx = QtWidgets.QCheckBox('Manufacturer correction')
        self.ManufacturerCorrectionChkBx.setChecked(True)
        self.zernikeCorrectionChkBx = QtWidgets.QCheckBox('Aberration control')
        self.zernikeCorrectionChkBx.setChecked(True)
        self.pupilLabel = QtWidgets.QLabel('PUPIL')
        self.pupilLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.pupilChkBx = QtWidgets.QCheckBox('radius (mm):')
        self.pupilChkBx.setChecked(False)
        self.pupilEdt = MyQtLineEdit('1')
        self.pupilEdt.setFixedWidth(100)
        self.pupilCenterLabel = QtWidgets.QLabel('PUPIL CENTER (mm)[x0, y0]:')
        self.pupilCenterEdt_x0 = MyQtLineEdit(str(0))
        self.pupilCenterEdt_y0 = MyQtLineEdit(str(0))
        self.pupilCenterEdt_x0.setFixedWidth(50)
        self.pupilCenterEdt_y0.setFixedWidth(50)
        self.pupilOutLabel = QtWidgets.QLabel('Move out')
        self.pupilOutLineEdit = MyQtLineEdit('40')
        self.pupilOutLineEdit.setFixedWidth(50)
        # self.pupilTypeCB = QtWidgets.QComboBox()
        # self.pupilTypeCB.addItems(['circular','square','rectangular','rectangular_rotated', 'rectangular_rotated_byAtom'])
        # self.ovulaLabel = QtWidgets.QLabel('OVULA TRAP')
        # self.ovulaLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.ovulaTrapChkBx = QtWidgets.QCheckBox('Disk radius (mm) :')
        # self.ovulaTrapChkBx.setChecked(False)
        # self.sizeRingEdt = MyQtLineEdit('2.5')
        # self.sizeRingEdt.setFixedWidth(50)
        # self.tem01Label = QtWidgets.QLabel('TEM01')
        # self.tem01Label.setAlignment(QtCore.Qt.AlignCenter)
        # self.tem01ChkBx = QtWidgets.QCheckBox('Spacing (in um) :')
        # self.tem01ChkBx.setChecked(False)
        # self.tem01spacingEdt = MyQtLineEdit(str(self.spacing))
        # self.tem01spacingEdt.setFixedWidth(50)
        # self.tem01PfactorLabel = QtWidgets.QLabel('P factor (in pi, default=1) :')
        # self.tem01PfactorEdt = MyQtLineEdit('1')
        # self.tem01PfactorEdt.setFixedWidth(50)
        # self.tem01ThetaLabel = QtWidgets.QLabel('Angle (in degree, default=0) :')
        # self.tem01ThetaEdt = MyQtLineEdit(str(self.theta_deg) )
        # self.tem01ThetaEdt.setFixedWidth(50)
        # self.SOLabel = QtWidgets.QLabel('Superoscillation')
        # self.SOLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.SOChkBx = QtWidgets.QCheckBox('r_pi [from 0 to 1]:')
        # self.SOChkBx.setChecked(False)
        # self.SOrpiEdt = MyQtLineEdit(str(self.r_pi))
        # self.SOrpiEdt.setFixedWidth(50)
        # self.SOPfactorLabel = QtWidgets.QLabel('P factor (in pi, default=1) :')
        # self.SOPfactorEdt = QtWidgets.QLineEdit('1')
        # self.SOPfactorEdt.setFixedWidth(50)
        # self.SOCenterLabel = QtWidgets.QLabel('SO CENTER (mm)[x0, y0]:')
        # self.SOCenterEdt_x0 = QtWidgets.QLineEdit(str(0))
        # self.SOCenterEdt_y0 = QtWidgets.QLineEdit(str(0))
        # self.SOCenterEdt_x0.setFixedWidth(50)
        # self.SOCenterEdt_y0.setFixedWidth(50)
        # self.LGLabel = QtWidgets.QLabel('Laguerre-Gaussian')
        # self.LGLabel.setAlignment(QtCore.Qt.AlignCenter)
        # self.LGChkBx = QtWidgets.QCheckBox('LG01')
        # self.LGChkBx.setChecked(False)
        # self.LGPfactorLabel = QtWidgets.QLabel('P factor (in pi, default=1) :')
        # self.LGPfactorEdt = QtWidgets.QLineEdit('1')
        # self.LGPfactorEdt.setFixedWidth(50)
        # self.LGCenterLabel = QtWidgets.QLabel('LG CENTER (mm)[x0, y0]:')
        # self.LGCenterEdt_x0 = QtWidgets.QLineEdit(str(0))
        # self.LGCenterEdt_y0 = QtWidgets.QLineEdit(str(0))
        # self.LGCenterEdt_x0.setFixedWidth(50)
        # self.LGCenterEdt_y0.setFixedWidth(50)

        self.Advanced_control_layout.addWidget(self.label_Advanced_control_layout, row=0, colspan=3)
        self.Advanced_control_layout.addWidget(self.correctionLabel, row=23, col=0)
        self.Advanced_control_layout.addWidget(self.ManufacturerCorrectionChkBx, row=23, col=1)
        self.Advanced_control_layout.addWidget(self.zernikeCorrectionChkBx, row=23, col=2)
        self.Advanced_control_layout.addWidget(self.pupilLabel, row=25, col=0)
        self.Advanced_control_layout.addWidget(self.pupilChkBx, row=25, col=1)
        self.Advanced_control_layout.addWidget(self.pupilEdt, row=25, col=2)
        # self.Advanced_control_layout.addWidget(self.pupilTypeCB, row=25, col=3)
        self.Advanced_control_layout.addWidget(self.pupilCenterLabel, row=26, col=0)
        self.Advanced_control_layout.addWidget(self.pupilCenterEdt_x0, row=26, col=1)
        self.Advanced_control_layout.addWidget(self.pupilCenterEdt_y0, row=26, col=2)
        self.Advanced_control_layout.addWidget(self.pupilOutLabel, row=28, col=1)
        self.Advanced_control_layout.addWidget(self.pupilOutLineEdit, row=28, col=2)
        # self.Advanced_control_layout.addWidget(self.ovulaLabel, row=35, col=0)
        # self.Advanced_control_layout.addWidget(self.ovulaTrapChkBx, row=35, col=1)
        # self.Advanced_control_layout.addWidget(self.sizeRingEdt, row=35, col=2)
        # self.Advanced_control_layout.addWidget(self.tem01Label, row=45, col=0)
        # self.Advanced_control_layout.addWidget(self.tem01ChkBx, row=45, col=1)
        # self.Advanced_control_layout.addWidget(self.tem01spacingEdt, row=45, col=2)
        # self.Advanced_control_layout.addWidget(self.tem01PfactorLabel, row=45, col=3)
        # self.Advanced_control_layout.addWidget(self.tem01PfactorEdt, row=45, col=4)
        # self.Advanced_control_layout.addWidget(self.tem01ThetaLabel, row=47, col=1)
        # self.Advanced_control_layout.addWidget(self.tem01ThetaEdt, row=47, col=2)
        # self.Advanced_control_layout.addWidget(self.SOLabel, row=55, col=0)
        # self.Advanced_control_layout.addWidget(self.SOChkBx, row=55, col=1)
        # self.Advanced_control_layout.addWidget(self.SOrpiEdt, row=55, col=2)
        # self.Advanced_control_layout.addWidget(self.SOPfactorLabel, row=55, col=3)
        # self.Advanced_control_layout.addWidget(self.SOPfactorEdt, row=55, col=4)
        # self.Advanced_control_layout.addWidget(self.SOCenterLabel, row=56, col=1)
        # self.Advanced_control_layout.addWidget(self.SOCenterEdt_x0, row=56, col=2)
        # self.Advanced_control_layout.addWidget(self.SOCenterEdt_y0, row=56, col=3)
        # self.Advanced_control_layout.addWidget(self.LGLabel, row=65, col=0)
        # self.Advanced_control_layout.addWidget(self.LGChkBx, row=65, col=1)
        # self.Advanced_control_layout.addWidget(self.LGPfactorLabel, row=65, col=3)
        # self.Advanced_control_layout.addWidget(self.LGPfactorEdt, row=65, col=4)
        # self.Advanced_control_layout.addWidget(self.LGCenterLabel, row=66, col=1)
        # self.Advanced_control_layout.addWidget(self.LGCenterEdt_x0, row=66, col=2)
        # self.Advanced_control_layout.addWidget(self.LGCenterEdt_y0, row=66, col=3)

        self.ManufacturerCorrectionChkBx.stateChanged.connect(self.gui.updatePhase)
        self.zernikeCorrectionChkBx.stateChanged.connect(self.gui.updatePhase)
        self.pupilChkBx.stateChanged.connect(self.calculatePupil)
        self.pupilEdt.returnPressed.connect(self.calculatePupil)
        self.pupilCenterEdt_x0.returnPressed.connect(self.calculatePupil)
        self.pupilCenterEdt_y0.returnPressed.connect(self.calculatePupil)
        self.pupilOutLineEdit.returnPressed.connect(self.update_k_dump)
        # self.ovulaTrapChkBx.stateChanged.connect(self.createOvulaDisk)
        # self.sizeRingEdt.returnPressed.connect(self.createOvulaDisk)
        # self.tem01ChkBx.stateChanged.connect(lambda : self.createTEM01trap())
        # self.tem01spacingEdt.returnPressed.connect(self.createTEM01trap)
        # self.tem01PfactorEdt.returnPressed.connect(self.createTEM01trap)
        # self.tem01ThetaEdt.returnPressed.connect(self.createTEM01trap_angle)
        # self.SOChkBx.stateChanged.connect(lambda : self.createSOtrap())
        # self.SOrpiEdt.returnPressed.connect(self.createSOtrap)
        # self.SOPfactorEdt.returnPressed.connect(self.createSOtrap)
        # self.SOCenterEdt_x0.returnPressed.connect(self.createSOtrap)
        # self.SOCenterEdt_y0.returnPressed.connect(self.createSOtrap)
        # self.LGChkBx.stateChanged.connect(lambda : self.createLGtrap())
        # self.LGPfactorEdt.returnPressed.connect(self.createLGtrap)
        # self.LGCenterEdt_x0.returnPressed.connect(self.createLGtrap)
        # self.LGCenterEdt_y0.returnPressed.connect(self.createLGtrap)

    def update_k_dump(self):
        d_out = float(self.gui.pupilOutLineEdit.text())
        self.gui.k_dump = np.array([d_out,0])*1e-6*position_to_step_per_pixel

    def calculatePupil(self):
        d_out = float(self.gui.pupilOutLineEdit.text())
        dphi_x = 2*np.pi/param.lamb * param.M/param.f * d_out*1e-6
        dphi_y = 2*np.pi/param.lamb * param.M/param.f * d_out*1e-6
        self.gui.grating_out = self.gui.X*dphi_x + self.gui.Y*dphi_y

        pupil_x0 = float(self.gui.pupilCenterEdt_x0.text())*1e-3
        pupil_y0 = float(self.gui.pupilCenterEdt_y0.text())*1e-3
        # self.gui.pupil_type = str(self.pupilTypeCB.currentText())
        self.gui.pupil_type = 'circular'
        self.gui.pupil = np.zeros((param.ImgResY,param.ImgResX)).astype('float64')

        if self.gui.pupil_type == 'circular':
            r_pupil = float(self.gui.pupilEdt.text())*1e-3

            self.gui.pupil[((self.gui.X - pupil_x0)**2 + (self.gui.Y - pupil_y0)**2)< r_pupil**2] = 1.
            self.gui.grating_out[((self.gui.X - pupil_x0)**2 + (self.gui.Y - pupil_y0)**2)< r_pupil**2] = 0.

        # elif self.gui.pupil_type == 'rectangular':
        #     txt = self.gui.pupilEdt.text()
        #     width, height = np.array(txt.split(','),dtype = 'float')*1e-3

        #     cut_select = np.logical_and(np.abs(self.gui.X)< width,np.abs(self.gui.Y)<height)
        #     self.gui.pupil[cut_select] = 1.
        #     self.gui.grating_out[cut_select] = 0.

        # elif self.gui.pupil_type == 'rectangular_rotated':
        #     txt = self.gui.pupilEdt.text()
        #     width, height, θ = np.array(txt.split(','),dtype = 'float')
        #     width *= 1e-3
        #     height *=1e-3
        #     θ *= np.pi/180
        #     x_rot = np.cos(θ)*self.gui.X + np.sin(θ)*self.gui.Y
        #     y_rot = np.sin(θ)*self.gui.X - np.cos(θ)*self.gui.Y

        #     cut_select = np.logical_and(np.abs(x_rot)< width,np.abs(y_rot)<height)
        #     self.gui.pupil[cut_select] = 1.
        #     self.gui.grating_out[cut_select] = 0.

        # elif self.gui.pupil_type == 'square':
        #     r_pupil = float(self.gui.pupilEdt.text())*1e-3

        #     self.gui.pupil[np.logical_and(np.abs(self.gui.X)< r_pupil,np.abs(self.gui.Y)<r_pupil)] = 1.
        #     self.gui.grating_out[np.logical_and(np.abs(self.gui.X)< r_pupil, np.abs(self.gui.Y)<r_pupil)] = 0.

        # elif self.gui.pupil_type == 'rectangular_rotated_byAtom':
        #     self.gui.pupil = np.ones((param.ImgResY,param.ImgResX)).astype('float64')

        self.gui.updatePhase()


    def applyPupil(self):
        '''
        Create an artificial pupil limiting the NA of the trap.

        Inside a radius defined by the user in self.gui.pupilEdt
        the phase pattern is not modified

        Outside this radius, the phase pattern is replaced by a grating with
        a large (user defined) wavevector to displace the light in this region
        '''

        # self.gui.pupil_type = str(self.pupilTypeCB.currentText())

        # if self.gui.pupil_type == 'rectangular_rotated_byAtom':
        #     """
        #     Apply pupil by atoms.
        #     """
        #     if not self._ensure_traps_loaded():
        #         print('applyPupil (by atom): falling back to standard pupil; trap load failed.')
        #         self.gui.totalPhase *= self.gui.pupil
        #         self.gui.totalPhase += self.gui.grating_out
        #         return

        #     #Chew
        #     txt = self.gui.pupilEdt.text()
        #     width, height, θ = np.array(txt.split(','),dtype = 'float')
        #     print('original long axis')
        #     print(self.gui.pupil_traps['pupil_longaxis'])

        #     long_axis = self.gui.pupil_traps['pupil_longaxis'] * width
        #     short_axis = self.gui.pupil_traps['pupil_shortaxis'] * height
        #     pupil_angle = self.gui.pupil_traps['pupil_angle']

        #     print('after long axis')
        #     print(self.gui.pupil_traps['pupil_longaxis']* width)

        #     pupil_traps = {
        #         'pupil_longaxis': long_axis,
        #         'pupil_shortaxis': short_axis,
        #         'pupil_angle': pupil_angle,
        #         }

        #     phi = trap_phase_to_slm(
        #         self.gui.traps_k,
        #         self.gui.theta_traps,
        #         self.gui.weight_traps,
        #         pupil = pupil_traps,
        #         shape = self.gui.SLM_shape,
        #         k_dump = self.gui.k_dump
        #         )
        #     self.gui.totalPhase = self.gui.totalPhase - self.gui.phasePattern + phi

        # else:
        self.gui.totalPhase *= self.gui.pupil
        self.gui.totalPhase += self.gui.grating_out


    def apply_pupil_longaxis_byatom_loop(self, longaxis_ratio = 1.0):
        txt = self.gui.pupilEdt.text()
        width, height, θ = np.array(txt.split(','),dtype = 'float')

        width = longaxis_ratio

        new_pupil = str(width) + ',' + str(height) + ',' + str(θ)
        self.gui.pupilEdt.setText(new_pupil)

        self.gui.updatePhase()

    def apply_pupil_shortaxis_byatom_loop(self, shortaxis_ratio = 1.0):
        txt = self.gui.pupilEdt.text()
        width, height, θ = np.array(txt.split(','),dtype = 'float')

        height = shortaxis_ratio

        new_pupil = str(width) + ',' + str(height) + ',' + str(θ)
        self.gui.pupilEdt.setText(new_pupil)

        self.gui.updatePhase()

    def createOvulaDisk(self):
        '''
        Make the so-called ovula trap

        Shift the phase by pi in an inner disk of radius defined by the user
        '''
        r = float(self.sizeRingEdt.text())/1000

        phi_ovula = np.zeros((param.ImgResY,param.ImgResX))
        for i in range(param.ImgResY):
            for j in range(param.ImgResX):
                if ((self.gui.X[i,j]**2 + self.gui.Y[i,j]**2)< r**2):
                    phi_ovula[i,j+PUPIL_VERTICAL_OFFSET] = np.pi
                else:
                    phi_ovula[i,j+PUPIL_VERTICAL_OFFSET] = 0

        self.gui.phi_ovula = phi_ovula
        self.gui.updatePhase()

    def createTEM01trap(self, value=None):
        '''
        Make the TEM01-like traps
        Make a patten with a 0-pi phase difference between upper and lower side
        '''
        if type(value) == float or type(value) == int:
            self.gui.spacing = value
            self.tem01spacingEdt.setText(f'{self.gui.spacing: .3}')
        else:
            self.gui.spacing = float(self.tem01spacingEdt.text())

        phi_tem01 = np.zeros((param.ImgResY,param.ImgResX))
        P_factor = float(self.tem01PfactorEdt.text())

        self.gui.theta_deg = float(self.tem01ThetaEdt.text())
        theta_rad = self.gui.theta_deg/180*np.pi
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        amp_list = np.sin(-2*np.pi*0.5*self.gui.spacing*1e-6*(np.cos(theta_rad)*Y+np.sin(theta_rad)*X)/(param.lamb*param.f/param.M))
        phase_list = np.array([amp_list >= 0])*np.pi*P_factor
        phi_tem01 = phase_list[0,:,:]

        self.gui.phi_tem01 = phi_tem01
        self.gui.updatePhase()

    def createTEM01trap_angle(self, value=None):
        '''
        Make the TEM01-like traps(change the angle)

        '''

        if type(value) == float or type(value) == int:
            self.gui.theta_deg = value
            self.tem01ThetaEdt.setText(f'{self.gui.theta_deg: .3}')

        else:
            self.gui.theta_deg = float(self.tem01ThetaEdt.text()) #[m]

        self.gui.spacing = float(self.tem01spacingEdt.text())
        phi_tem01 = np.zeros((param.ImgResY,param.ImgResX))
        #P_factor = 1
        P_factor = float(self.tem01PfactorEdt.text())
        #theta_deg = -45
        theta_rad = self.gui.theta_deg/180*np.pi
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        amp_list = np.sin(-2*np.pi*0.5*self.gui.spacing*1e-6*(np.cos(theta_rad)*Y+np.sin(theta_rad)*X)/(param.lamb*param.f/param.M))
        phase_list = np.array([amp_list >= 0])*np.pi*P_factor
        phi_tem01 = phase_list[0,:,:]

        self.gui.phi_tem01 = phi_tem01
        self.gui.updatePhase()

    def createSOtrap(self, value=None):
        '''
        Make the superoscillation traps
        Make a circular patten with a 0-pi phase difference between center and outer region
        '''
        if type(value) == float or type(value) == int:
            self.gui.r_pi = value
            self.SOrpiEdt.setText(f'{self.gui.r_pi:.2f}')

        else:
            self.gui.r_pi = float(self.SOrpiEdt.text()) #[unitless]

        SO_x0 = float(self.SOCenterEdt_x0.text())*1e-3 #[m]
        SO_y0 = float(self.SOCenterEdt_y0.text())*1e-3 #[m]
        r_pi = self.gui.r_pi
        phi_SO = np.zeros((param.ImgResY,param.ImgResX))
        P_factor = float(self.SOPfactorEdt.text())
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        Rho = ((X-SO_x0)**2 + (Y-SO_y0)**2)**0.5
        phase_list = np.array([Rho <= r_pi*param.ImgResY/2*param.pslm])*np.pi*P_factor
        phi_SO = phase_list[0,:,:]

        self.gui.phi_SO = phi_SO
        self.gui.updatePhase()

    def createLGtrap(self):
        '''
        Make the Laguerre-Gaussian 01 mode traps.
        Make a spiral phase patten from 0 to 2 pi corresponding to the polar angle.
        '''

        LG_x0 = float(self.LGCenterEdt_x0.text())*1e-3 #[m]
        LG_y0 = float(self.LGCenterEdt_y0.text())*1e-3 #[m]

        phi_LG = np.zeros((param.ImgResY,param.ImgResX))
        P_factor = float(self.LGPfactorEdt.text())
        x = np.arange(-param.ImgResX/2,param.ImgResX/2)*param.pslm
        y = np.arange(-param.ImgResY/2,param.ImgResY/2)*param.pslm
        X,Y = np.meshgrid(x,y)
        Theta = np.arctan2((Y-LG_y0),(X-LG_x0)) + np.pi
        phase_list = Theta*P_factor
        phi_LG = phase_list#[0,:,:]

        self.gui.phi_LG = phi_LG
        self.gui.updatePhase()

    def generateLGMask(self):
        '''
        Generates the phase mask for Laguerre-Gauss beams

        CURRENTLY NOT ACCESSIBLE FROM THE GUI
        '''
        self.gui.angularM = 1
        self.gui.ImgCenterX = 400
        self.gui.ImgCenterY = 300

        if self.gui.angularM == 0:
            image = np.zeros([param.ImgResY, param.ImgResX])
        else:
            image = np.angle(np.exp((self.gui.angularM*np.angle((self.gui.X - self.gui.ImgCenterX) + (self.gui.Y - self.gui.ImgCenterY)*1j))*1j))

        return image
