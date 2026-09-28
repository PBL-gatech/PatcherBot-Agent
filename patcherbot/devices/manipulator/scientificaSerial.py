
import serial
from .manipulator import Manipulator
import time
import threading
import re

__all__ = ['ScientificaSerial']

class SerialCommands():
    GET_X_POS = 'PX\r'
    GET_Y_POS = 'PY\r'
    GET_Z_POS = 'PZ\r'
    GET_X_Y_Z = '\r'
    GET_MAX_SPEED = 'TOP\r'
    GET_MAX_SPEED_Z = 'TOPZ\r'
    GET_MAX_ACCEL = 'ACC\r'
    GET_MAX_ACCEL_Z = 'ACCZ\r'
    GET_IS_BUSY = 's\r'

    SET_X_Y_POS_ABS = 'abs {} {}\r'
    SET_X_Y_POS_REL = 'rel {} {}\r'
    SET_X_Y_Z_POS_ABS = 'abs {} {} {}\r'
    SET_X_Y_Z_POS_REL = 'rel {} {} {}\r'
    

    SET_Z_POS = 'absz {}\r'
    SET_MAX_SPEED = 'TOP {}\r'
    SET_MAX_SPEED_Z = 'TOPZ {}\r'
    SET_MAX_ACCEL = 'ACC {}\r'
    SET_MAX_ACCEL_Z = 'ACCZ {}\r'

    SET_X_Y_Z_VEL = 'VJ {} {} {}\r'

    GET_BAUD = 'BAUD\r'
    SET_BAUD = 'BAUD {}\r'

    STOP = 'STOP\r'
    SET_OBJECTIVE = 'OBJ {}\r'


_number_re = re.compile(r'-?\d+')


def _parse_scientifica_int(response):
    if response is None:
        return None
    resp = str(response).strip()
    if not resp or resp.startswith('E,'):
        return None
    match = _number_re.search(resp)
    if match is None:
        return None
    return int(match.group(0))


class EncoderCorrectionAcquisitionThread(threading.Thread):
    """Background encoder + stage polling for ScientificaSerialEncoder."""

    def __init__(self, parent, z_axis_port, polling_freq):
        super().__init__(daemon=True, name="encoder_correction_thread")
        self._parent = parent
        self._z_axis_port = z_axis_port
        self._polling_freq = polling_freq
        self._encoder_seq = 0
        self._encoder_lock = threading.Lock()
        self._encoder_buffer = bytearray()
        self._encoder_expect_value = False
        self._stage_pos = [0, 0, 0]

    def run(self):
        self.run_loop()

    def run_loop(self, freq=None):
        if freq is None:
            freq = self._polling_freq
        while True:
            start_time = time.time()
            try:
                self._poll_encoder_stream()
            except Exception:
                pass

            try:
                xyz = self._parent._sendCmd(SerialCommands.GET_X_Y_Z)
                xyz = xyz.split('\t')
                x_pos = int(xyz[0]) / 10.0
                y_pos = int(xyz[1]) / 10.0
                z_pos = int(xyz[2]) / 10.0
                self._stage_pos = [x_pos, y_pos, z_pos]
            except Exception:
                print('error reading position')

            encoder_z, _ = self.get_encoder_state()
            self._parent.current_pos = [self._stage_pos[0], self._stage_pos[1], encoder_z]

            sleep_time = 1 / freq - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_encoder_state(self):
        with self._encoder_lock:
            return self._parent.encoderZ, self._encoder_seq

    def get_stage_z(self):
        return self._stage_pos[2]

    def wait_for_update(self, last_seq, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            _, seq = self.get_encoder_state()
            if seq != last_seq:
                return True
            time.sleep(0.01)
        return False

    def _update_encoder_from_count(self, count):
        with self._encoder_lock:
            self._parent.encoderZ = count * self._parent.stageUnitsPerEncoderPulse
            self._encoder_seq += 1

    def _consume_encoder_bytes(self, data):
        if not data:
            return
        self._encoder_buffer.extend(data)
        while b'\n' in self._encoder_buffer:
            line, _, remainder = self._encoder_buffer.partition(b'\n')
            self._encoder_buffer = remainder
            line = line.strip()
            if not line:
                continue
            if line == b'ENC':
                self._encoder_expect_value = True
                continue
            if self._encoder_expect_value:
                self._encoder_expect_value = False
                try:
                    count = int(line.decode('ascii', errors='ignore').strip())
                except ValueError:
                    continue
                self._update_encoder_from_count(count)
                continue

            # Backward compatibility: support plain integer line streams.
            try:
                count = int(line.decode('ascii', errors='ignore').strip())
            except ValueError:
                continue
            self._update_encoder_from_count(count)

    def _poll_encoder_stream(self):
        try:
            waiting = self._z_axis_port.in_waiting
        except Exception:
            return
        if not waiting:
            return
        try:
            data = self._z_axis_port.read(waiting)
        except Exception:
            return
        self._consume_encoder_bytes(data)


class ScientificaSerialEncoder(Manipulator):
    DEFAULT_STAGE_UNITS_PER_ENCODER_PULSE = 1.45
    DEFAULT_MAX_SPEED = 10000
    DEFAULT_MAX_ACCEL = 100
    DEFAULT_POLLING_FREQ = 10
    DEFAULT_Z_CORRECTION_TOLERANCE_UM = 2.0
    DEFAULT_Z_CORRECTION_MAX_RETRIES = 5

    def __init__(self, comPort: serial.Serial, zAxisComPort,
                 stage_units_per_encoder_pulse=None,
                 max_speed=None,
                 max_accel=None,
                 polling_freq=None,
                 stageUnitsPerEncoderPulse=None,
                 objective_lift_um=None):
        self.comPort : serial.Serial = comPort

        self.zAxisComPort : serial.Serial = zAxisComPort
        if stageUnitsPerEncoderPulse is not None and stage_units_per_encoder_pulse is None:
            stage_units_per_encoder_pulse = stageUnitsPerEncoderPulse
        self.stageUnitsPerEncoderPulse = (
            self.DEFAULT_STAGE_UNITS_PER_ENCODER_PULSE
            if stage_units_per_encoder_pulse is None
            else stage_units_per_encoder_pulse
        )
        self.encoderZ = 0

        self._lock = threading.Lock()
        self._supports_stage_z_profile = None
        self.current_pos = [0, 0, 0]
        self._current_objective = 1
        self._polling_freq = self.DEFAULT_POLLING_FREQ if polling_freq is None else polling_freq

        # self.info(f"Baud Rate: {self.get_baud_rate()}")

        self.set_max_accel(self.DEFAULT_MAX_ACCEL if max_accel is None else max_accel)
        self.set_max_speed(self.DEFAULT_MAX_SPEED if max_speed is None else max_speed)
        self.objective_lift_um = 10000.0 if objective_lift_um is None else float(objective_lift_um)

        self._encoder_acq = EncoderCorrectionAcquisitionThread(
            parent=self,
            z_axis_port=self.zAxisComPort,
            polling_freq=self._polling_freq,
        )
        self._polling_thread = self._encoder_acq
        self._polling_thread.start()

    def get_baud_rate(self):
        '''
        gets the baud rate of the serial port

        '''
        resp = self._sendCmd(SerialCommands.GET_BAUD)
        return (resp)

    def set_baud_rate(self, baud_rate : int):
        '''Sets the baud rate of the serial port.  
        '''
        self._sendCmd(SerialCommands.SET_BAUD.format(int(baud_rate)))

    def _get_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return None
        resp = self._sendCmd(command)
        value = _parse_scientifica_int(resp)
        if value is None and str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
            return None
        if value is not None:
            self._supports_stage_z_profile = True
        return value

    def _set_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return
        resp = self._sendCmd(command)
        if str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
        else:
            self._supports_stage_z_profile = True

    def get_max_speed(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_SPEED)
        xy_speed = _parse_scientifica_int(xy_resp)
        if xy_speed is None:
            self.warning(f"Scientifica TOP read failed: {xy_resp}")
            return None
        z_speed = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_SPEED_Z)
        return min(xy_speed, z_speed) if z_speed is not None else xy_speed

    def get_max_accel(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_ACCEL)
        xy_accel = _parse_scientifica_int(xy_resp)
        if xy_accel is None:
            self.warning(f"Scientifica ACC read failed: {xy_resp}")
            return None
        z_accel = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_ACCEL_Z)
        return min(xy_accel, z_accel) if z_accel is not None else xy_accel

    def set_max_speed(self, speed):
        speed = int(speed)
        resp = self._sendCmd(SerialCommands.SET_MAX_SPEED.format(speed))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica TOP set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_SPEED_Z.format(speed))

    def set_max_accel(self, accel):
        accel = int(accel)
        resp = self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(accel))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica ACC set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_ACCEL_Z.format(accel))


    def __del__(self):
        try:
            if hasattr(self, "comPort") and self.comPort:
                self.comPort.close()
        except Exception:
            pass

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response
        '''
        with self._lock:
            self.comPort.write(cmd.encode())
            resp = self.comPort.read_until(b'\r') #read reply to message
            resp = resp[:-1]

        return resp.decode()

    def position(self, axis=None):
        if axis == 1:
            return self.current_pos[0]
        if axis == 2:
            return self.current_pos[1]
        if axis == 3:
            if hasattr(self, "_encoder_acq"):
                encoder_z, _ = self._encoder_acq.get_encoder_state()
                return encoder_z
            return self.encoderZ
        if axis == None:
            return self.current_pos
        
    def update_pos_continuous(self, freq=None):
        '''constantly polls the device's position and updates the current_pos variable
        '''
        self._encoder_acq.run_loop(freq=freq)

    def absolute_move(self, pos, axis):
        print(f'[OBJDBG] {self.__class__.__name__}.absolute_move axis={axis} pos_um={pos}')

        if axis == 1:
            yPos = self.position(axis=2)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(pos * 10) , int(yPos * 10)))
        if axis == 2:
            xPos = self.position(axis=1)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(pos * 10)))
        if axis == 3:
            max_retries = self.DEFAULT_Z_CORRECTION_MAX_RETRIES
            tolerance = self.DEFAULT_Z_CORRECTION_TOLERANCE_UM
            for attempt in range(max_retries):
                stageZ = self._encoder_acq.get_stage_z() if hasattr(self, "_encoder_acq") else self.current_pos[2]
                encoder_z, seq = self._encoder_acq.get_encoder_state() if hasattr(self, "_encoder_acq") else (self.encoderZ, 0)
                setpointStage = stageZ + (pos - encoder_z)
                self._sendCmd(SerialCommands.SET_Z_POS.format(int(setpointStage * 10)))
                self.wait_until_still()
                time.sleep(1)
                if hasattr(self, "_encoder_acq"):
                    self._encoder_acq.wait_for_update(seq, max(0.1, 2.0 / self._polling_freq))
                    encoder_z, _ = self._encoder_acq.get_encoder_state()
                print(f'expected encoder: {pos} actual {encoder_z}')
                error = pos - encoder_z
                print(f'error: {error}')
                if abs(error) <= tolerance:
                    break
                if attempt < max_retries - 1:
                    print('retrying')
    
    def absolute_move_group(self, x, axes, speed=None):
        x = list(x)
        axes = list(axes)

        if 1 in axes and 2 in axes and 3 in axes:
            # Move X, Y and corrected Z together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            zPos = x[axes.index(3)]
            print("sent cmd", xPos, yPos, zPos)
            stageZ = self._encoder_acq.get_stage_z() if hasattr(self, "_encoder_acq") else self.current_pos[2]
            encoder_z, _ = self._encoder_acq.get_encoder_state() if hasattr(self, "_encoder_acq") else (self.encoderZ, 0)
            setpointStage = stageZ + (zPos - encoder_z)
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_ABS.format(int(xPos * 10), int(yPos * 10), int(setpointStage * 10)))

        elif 1 in axes and 2 in axes:
            # Move X and Y axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            print("sent cmd", xPos, yPos)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(yPos * 10)))

        else:
            print(f'unimplemented move group {x} {axes}')

    
    def relative_move_group(self, x, axes, speed=None):
        """
        Relative multi‑axis move using Scientifica's `rel` command.
        Mirrors absolute_move_group but sends deltas instead of targets.
        """
        x = list(x)
        axes = list(axes)

        # Build delta vector in device order (1=X, 2=Y, 3=Z)
        dx = dy = dz = 0
        if 1 in axes:
            dx = int(x[axes.index(1)] * 10)
        if 2 in axes:
            dy = int(x[axes.index(2)] * 10)
        if 3 in axes:
            dz = int(x[axes.index(3)] * 10)

        if 1 in axes and 2 in axes and 3 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(dx, dy, dz))
        elif 1 in axes and 2 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(dx, dy))
        elif 3 in axes:
            # Only Z move; still use the 3‑axis relative command for consistency
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(0, 0, dz))
        else:
            print(f'unimplemented move group {x} {axes}')

    def absolute_move_group_velocity(self,vel,axes):   
        try: 
         self.info(f"Setting velocity to {vel} on axes {axes}")
         vel = list(vel)
         axes = list(axes)
         xvel = vel[axes.index(1)]
         yvel = vel[axes.index(2)]
         zvel = vel[axes.index(3)]
         self._sendCmd(SerialCommands.SET_X_Y_Z_VEL.format(xvel, yvel, zvel))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")

    def relative_move_group_velocity(self, vel, axes):
        """
        Relative velocity-mode API; Scientifica firmware uses direct axis velocities,
        so this is equivalent to absolute_move_group_velocity.
        """
        self.absolute_move_group_velocity(vel, axes)

    def wait_until_still(self, axes = None, axis = None):
        while True:
            resp = self._sendCmd(SerialCommands.GET_IS_BUSY)
            busy = resp != '0'

            if not busy:
                break

    def stop(self):
        self._sendCmd(SerialCommands.STOP)

    def get_current_objective(self):
        return self._current_objective

    def switch_objective(self, target):
        try:
            target = int(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid objective target: {target}') from exc

        if target not in (1, 2):
            raise ValueError(f'Objective target must be 1 or 2, got {target}')

        print(f'[OBJDBG] {self.__class__.__name__}.switch_objective target={target}')
        try:
            print(f'[OBJDBG] {self.__class__.__name__}.pre-OBJ stage_pos={self.position()}')
        except Exception as exc:
            print(f'[OBJDBG] {self.__class__.__name__}.pre-OBJ stage_pos read failed: {exc}')
        resp = self._sendCmd(SerialCommands.SET_OBJECTIVE.format(target))
        resp_text = str(resp).strip()
        print(f'[OBJDBG] {self.__class__.__name__}.switch_objective response={resp_text}')
        if resp_text != 'A':
            if resp_text.startswith('E,'):
                raise RuntimeError(f'Scientifica OBJ {target} failed: {resp_text}')
            raise RuntimeError(
                f'Scientifica OBJ {target} returned unexpected response: {resp_text or "<empty>"}'
            )
        self.wait_until_still()
        try:
            print(f'[OBJDBG] {self.__class__.__name__}.post-OBJ stage_pos={self.position()}')
        except Exception as exc:
            print(f'[OBJDBG] {self.__class__.__name__}.post-OBJ stage_pos read failed: {exc}')
        self._current_objective = target
        print(f'[OBJDBG] {self.__class__.__name__}.switch_objective done current={self._current_objective}')

class ScientificaSerialNoEncoder(Manipulator):

    def __init__(self, comPort: serial.Serial, objective_lift_um=None):
        self.comPort : serial.Serial = comPort
        self._lock = threading.Lock()
        self._supports_stage_z_profile = None
        self.current_pos = [0, 0, 0]
        self._current_objective = 1

        self.set_max_accel(1000)
        self.set_max_speed(100000)
        self.objective_lift_um = 10000.0 if objective_lift_um is None else float(objective_lift_um)
        self.info(f"Maximum Speed: {self.get_max_speed()} um/s, "f"Maximum Acceleration: {self.get_max_accel()} um/s^2")

        #start constantly polling position in a new thread
        self._polling_thread = threading.Thread(target=self.update_pos_continuous, daemon=True)
        self._polling_thread.start()
        self._polling_thread.deamon = True




    def _get_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return None
        resp = self._sendCmd(command)
        value = _parse_scientifica_int(resp)
        if value is None and str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
            return None
        if value is not None:
            self._supports_stage_z_profile = True
        return value

    def _set_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return
        resp = self._sendCmd(command)
        if str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
        else:
            self._supports_stage_z_profile = True

    def get_max_speed(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_SPEED)
        xy_speed = _parse_scientifica_int(xy_resp)
        if xy_speed is None:
            self.warning(f"Scientifica TOP read failed: {xy_resp}")
            return None
        z_speed = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_SPEED_Z)
        return min(xy_speed, z_speed) if z_speed is not None else xy_speed

    def get_max_accel(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_ACCEL)
        xy_accel = _parse_scientifica_int(xy_resp)
        if xy_accel is None:
            self.warning(f"Scientifica ACC read failed: {xy_resp}")
            return None
        z_accel = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_ACCEL_Z)
        return min(xy_accel, z_accel) if z_accel is not None else xy_accel

    def set_max_speed(self, speed):
        speed = int(speed)
        resp = self._sendCmd(SerialCommands.SET_MAX_SPEED.format(speed))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica TOP set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_SPEED_Z.format(speed))

    def set_max_accel(self, accel):
        accel = int(accel)
        resp = self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(accel))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica ACC set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_ACCEL_Z.format(accel))

    def __del__(self):
        self.comPort.close()

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response
        '''
        with self._lock:
            # start  = time.perf_counter_ns()
            self.comPort.write(cmd.encode())
            resp = self.comPort.read_until(b'\r') #read reply to message
            resp = resp[:-1]
            # if resp == b'A':
            #     print(f"command received: {resp}")
            # end = time.perf_counter_ns()
            # print(f"Time taken to send command: {(end - start)/1e6} ms")
        return resp.decode()

    def position(self, axis=None):
        if axis == 1:
            return self.current_pos[0]
        if axis == 2:
            return self.current_pos[1]
        if axis == 3:
            return self.current_pos[2]
        if axis == None:
            return self.current_pos
        
    def update_pos_continuous(self, freq=100):
        '''constantly polls the device's position and updates the current_pos variable
        '''
        while True:
            startTime = time.time()
            xyz = self._sendCmd(SerialCommands.GET_X_Y_Z)
            xyz = xyz.split('\t')
            
            try:
                xPos = int(xyz[0]) / 10.0
                yPos = int(xyz[1]) / 10.0
                zPos = int(xyz[2]) / 10.0
                self.current_pos = [xPos, yPos, zPos]
            except:
                print('error reading position')

            sleepTime = 1 / freq - (time.time() - startTime)
            if sleepTime > 0:
                time.sleep(sleepTime)

    def absolute_move(self, pos, axis, speed=None):
        print(f'[OBJDBG] {self.__class__.__name__}.absolute_move axis={axis} pos_um={pos}')
        '''Moves the device to an absolute position in um.
        '''
        # print(f"absolute move {pos} {axis}")
        try: 
            if axis == 1:
                yPos = self.position(axis=2)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(round(pos * 10) , round(yPos * 10)))
            if axis == 2:
                xPos = self.position(axis=1)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(round(xPos * 10), round(pos * 10)))
            if axis == 3:
                self._sendCmd(SerialCommands.SET_Z_POS.format(round(pos * 10)))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")
    
    def absolute_move_group(self, x, axes, speed=None):
        '''
        Moves the device group of axes to position x.
        Parameters
        ----------
        axes : list of axis numbers
        x : target position in um (vector or list).
        '''
        # self.abort_if_requested()
        x = list(x)
        axes = list(axes)

        if 1 in axes and 2 in axes and 3 not in axes:
            # Move X and Y axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(yPos * 10)))

        elif 1 in axes and 2 in axes and 3 in axes:
            # Move X, Y and Z axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            zPos = x[axes.index(3)]
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_ABS.format(int(xPos * 10), int(yPos * 10), int(zPos * 10)))
            # end  = time.perf_counter_ns()
            # print(f"Time taken to move: {(end - start)/1e6} ms")


        else:
            print(f'unimplemented move group {x} {axes}')

    def absolute_move_group_velocity(self, vel):
        '''
        Moves the device in um/s.
        Parameters
        ----------
        vel : list of velocities for each axis
        '''
        # self.abort_if_requested()   
        try: 
            vel = list(vel)
            if len(vel) != 3:
                raise ValueError("Expected velocity list of length 3: [xvel, yvel, zvel]")
            xvel, yvel, zvel = vel
            self._sendCmd(SerialCommands.SET_X_Y_Z_VEL.format(xvel, yvel, zvel))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")

    def relative_move_group_velocity(self, vel, axes=None):
        """
        Relative velocity-mode API; backend command accepts direct axis velocities.
        """
        self.absolute_move_group_velocity(vel)
        
    def relative_move_group(self, x, axes, speed=None):
        """
        Relative multi‑axis move using the 2‑ or 3‑axis `rel` commands.
        Mirrors absolute_move_group but sends deltas instead of targets.
        """
        x = list(x)
        axes = list(axes)

        dx = dy = dz = 0
        if 1 in axes:
            dx = int(x[axes.index(1)] * 10)
        if 2 in axes:
            dy = int(x[axes.index(2)] * 10)
        if 3 in axes:
            dz = int(x[axes.index(3)] * 10)

        if 1 in axes and 2 in axes and 3 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(dx, dy, dz))
        elif 1 in axes and 2 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(dx, dy))
        elif 3 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(0, 0, dz))
        else:
            print(f'unimplemented move group {x} {axes}')

    def wait_until_still(self, axes = None, axis = None):
        while True:
            resp = self._sendCmd(SerialCommands.GET_IS_BUSY)
            busy = resp != '0'

            if not busy:
                break

    def stop(self):
        self._sendCmd(SerialCommands.STOP)

    def get_current_objective(self):
        return self._current_objective

    def switch_objective(self, target):
        try:
            target = int(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid objective target: {target}') from exc

        if target not in (1, 2):
            raise ValueError(f'Objective target must be 1 or 2, got {target}')

        print(f'[OBJDBG] {self.__class__.__name__}.switch_objective target={target}')
        try:
            print(f'[OBJDBG] {self.__class__.__name__}.pre-OBJ stage_pos={self.position()}')
        except Exception as exc:
            print(f'[OBJDBG] {self.__class__.__name__}.pre-OBJ stage_pos read failed: {exc}')
        resp = self._sendCmd(SerialCommands.SET_OBJECTIVE.format(target))
        resp_text = str(resp).strip()
        print(f'[OBJDBG] {self.__class__.__name__}.switch_objective response={resp_text}')
        if resp_text != 'A':
            if resp_text.startswith('E,'):
                raise RuntimeError(f'Scientifica OBJ {target} failed: {resp_text}')
            raise RuntimeError(
                f'Scientifica OBJ {target} returned unexpected response: {resp_text or "<empty>"}'
            )
        self.wait_until_still()
        try:
            print(f'[OBJDBG] {self.__class__.__name__}.post-OBJ stage_pos={self.position()}')
        except Exception as exc:
            print(f'[OBJDBG] {self.__class__.__name__}.post-OBJ stage_pos read failed: {exc}')
        self._current_objective = target
        print(f'[OBJDBG] {self.__class__.__name__}.switch_objective done current={self._current_objective}')
