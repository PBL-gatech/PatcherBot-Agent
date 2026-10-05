# coding=utf-8
'''
Access and control of Acquisition threads for the DAQ, and Pressure controller
'''
import numpy as np
from enum import Enum

from patcherbot.interface import TaskInterface, command, blocking_command
from patcherbot.devices.pressurecontroller.BasePressureController import PressureController
from patcherbot.devices.amplifier.amplifier import Amplifier
from patcherbot.devices.amplifier.DAQ import DAQ
from patcherbot.devices.laser import Laser
from PyQt5 import QtCore
import time


__all__ = ['GraphInterface']

class GraphInterface(TaskInterface):
    def __init__(
        self,
        amplifier: Amplifier,
        daq: DAQ,
        pressure: PressureController,
        recording_state_manager,
        laser=None,
    ):
        super().__init__()
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.recording_state_manager = recording_state_manager
        self.laser = laser
        self._laser_power = 0
        self._last_laser_wavelength = None

    @command(category='Pressure', 
              description='obtain current pressure value')
    def get_last_pressure(self):
        return self.pressure.get_last_acquisition()
    @command(category='Pressure',
              description='change pressure setpoint',
              default_arg=0)
    def set_pressure(self, pressure):
        self.execute(self.pressure.set_pressure, argument=pressure)

    @command(category='Pressure',
              description='get pressure setpoint')
    def get_pressure(self):
        return self.pressure.get_pressure()
    @command(category='Pressure',
                description='switch pressure on or off',
                default_arg=False) 
    def set_ATM(self, atm):
        self.execute(self.pressure.set_ATM,argument=atm)
    @command(category='Pressure',
                    description='obtain current pressure state') 
    def get_ATM(self):
            return self.pressure.get_ATM()
        
    @command(category='DAQ',
                description='get last Data from DAQ')
    def get_last_data(self):
        if self.daq.get_last_acquisition() is not None:
            return self.daq.get_last_acquisition()
        else:
            return None

    @command(category='DAQ',
                description='compute noise metrics from DAQ response')
    def get_noise_metrics(self, timeData=None, respData=None,
                          window_start=0.004, window_end=0.010,
                          avg_p2p_window=0.001):
        if not hasattr(self.daq, "compute_noise_metrics"):
            return None
        return self.daq.compute_noise_metrics(
            timeData=timeData,
            respData=respData,
            window_start=window_start,
            window_end=window_end,
            avg_p2p_window=avg_p2p_window,
        )

    @command(category='DAQ',
                description='get last optogenetic protocol data')
    def get_last_optogenetic_data(self):
        return getattr(self.daq, "optogenetic_protocol_data", None)
        
    @command(category='DAQ',
                description='obtain acquision mode')
    def getCellMode(self):
        return self.daq.getCellMode()
    
    @command(category='DAQ',
                description=' set the acquisition mode',
                default_arg=False)
    def setCellMode(self, cellMode):
        self.execute(self.daq.setCellMode, argument=cellMode)

    @command(category = 'Amplifier',
                      description='set the Zap duration',
                      default_arg=0.5)
    def set_zap_duration(self, duration):
        self.execute(self.amplifier.set_zap_duration, argument=duration)

    @command(category = 'Amplifier',
                      description='Zap the cell',
                      success_message ='Zap done')

    def zap(self):
        self.execute(self.amplifier.zap)

    @command(category='Laser',
              description='get laser power state')
    def get_laser_power_state(self):
        return self.laser.get_power_state()

    @command(category='Laser',
              description='get current laser wavelength')
    def get_laser_wavelength(self):
        wavelength = self.laser.get_wavelength()
        self._last_laser_wavelength = wavelength
        return wavelength

    @command(category='Laser',
              description='get cached laser power percent')
    def get_laser_power(self):
        return self._laser_power

    @command(category='Laser',
              description='set laser power percent',
              default_arg=0)
    def set_laser_power(self, power_percent):
        try:
            power = int(round(float(power_percent)))
        except (TypeError, ValueError):
            self.warning("Invalid laser power input.")
            return None
        power = max(0, min(100, power))
        self.execute(self.laser.set_power_level, argument=power)
        self._laser_power = power
        return power

    @command(category='Laser',
              description='toggle laser output')
    def toggle_laser_output(self):
        power_state = self.laser.get_power_state()
        if power_state != "on":
            self.execute(self.laser.set_power_level, argument=self._laser_power)
        self.execute(self.laser.excite)
        return self.laser.get_power_state()

    @command(category='Laser',
              description='get available laser wavelength options')
    def get_laser_wavelength_options(self):
        order = getattr(self.laser, "_wavelength_order", None)
        if order:
            return list(order)
        current = self.laser.get_wavelength()
        if isinstance(current, Enum):
            return [c for c in type(current) if getattr(c, "name", "") != "OFF"]
        return []

    def _step_laser_wavelength(self, step: int):
        current = self.laser.get_wavelength()
        if current is None:
            target = 1
        elif isinstance(current, Enum):
            channels = [c for c in type(current) if getattr(c, "name", "") != "OFF"]
            if not channels:
                self.warning("Laser wavelength enum has no selectable channels.")
                return None
            try:
                idx = channels.index(current)
            except ValueError:
                idx = 0
            new_idx = max(0, min(len(channels) - 1, idx + step))
            target = channels[new_idx]
        elif isinstance(current, int):
            target = max(1, current + step)
        else:
            self.warning(f"Unsupported laser wavelength type: {type(current)}")
            return None

        self.execute(self.laser.set_wavelength, argument=target)
        self._last_laser_wavelength = target
        self.execute(self.laser.set_power_level, argument=self._laser_power)
        return target

    @command(category='Laser',
              description='step wavelength down')
    def wavelength_down(self):
        return self._step_laser_wavelength(-1)

    @command(category='Laser',
              description='step wavelength up')
    def wavelength_up(self):
        return self._step_laser_wavelength(1)

    



