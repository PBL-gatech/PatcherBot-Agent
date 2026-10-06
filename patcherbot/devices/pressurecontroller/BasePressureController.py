'''
A general pressure controller class with built-in acquisition methods.
'''
import collections
import threading
import time

from patcherbot.controller.base import TaskController

__all__ = ['PressureController', 'FakePressureController']


class PressureAcquisitionThread(threading.Thread):
    """
    A thread that continuously measures pressure via the controller's measure() method.
    The latest measurement is stored (using a deque with maxlen=1), and an optional
    callback is invoked with each new measurement.
    """
    def __init__(self, controller, interval=0.016, callback=None):
        """
        Initializes the acquisition thread.

        Args:
            controller (PressureController): Controller to measure pressure from.
            interval (float, optional): Sampling interval in seconds. Defaults to 0.016.
            callback (callable, optional): Optional callback for each measurement.
        """
        super().__init__(daemon=True)
        self.controller = controller
        self.interval = interval
        self.callback = callback
        self.running = True
        self._last_data_queue = collections.deque(maxlen=1)

    def run(self):
        """
        Runs the acquisition loop until stopped. Measures pressure and optionally
        invokes the callback.
        """
        while self.running:
            try:
                # Call the measure() method with no port parameter.
                pressure = self.controller.measure()
                # Ensure the measurement is an integer (as used in the GUI).
                pressure = int(pressure)
                # Publish the value and its acquisition time as one sample.
                self._last_data_queue.append((pressure, time.monotonic()))
                if self.callback:
                    self.callback(pressure)
            except Exception as e:
                self.controller.error("Error in PressureAcquisitionThread: " + str(e))
            time.sleep(self.interval)

    def get_last_data(self):
        """
        Returns the most recent measurement.

        Returns:
            int or None: Last pressure measurement, or None if unavailable.
        """
        #second return is the big update change, unclear during the merge which one of these needs to be used
        #return self._last_data_queue[-1] if self._last_data_queue else None
        return self._last_data_queue[-1][0] if self._last_data_queue else None

    def stop(self):
        self.running = False


class PressureController(TaskController):
    """
    Base class for pressure controllers with built-in acquisition capabilities.
    """
    def __init__(self):
        """
        Initializes the pressure controller.
        """
        super().__init__()
        self._pressure = collections.defaultdict(int)
        # Holder for the acquisition thread; child classes can start it.
        self._pressure_acq_thread = None
        self.state = False  # True if the pressure controller is in ATM mode, False otherwise

    def measure(self, port=0):
        """
        Measures the instantaneous pressure on the designated port.
        Child classes must override this method with device-specific behavior.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.
        
        Note: This method must be overriden in sublcass.
        """
        pass

    def set_pressure(self, pressure: int, port=0):
        """
        Sets the pressure on the designated port.

        Args:
            pressure (int): Desired pressure.
            port (int, optional): Pressure port number. Defaults to 0.
        """
        self._pressure[port] = pressure

    def get_pressure(self, port=0):
        """
        Gets the pressure value (as set via set_pressure) on the designated port.
        Note that this is not a measured value.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.

        Returns:
            int: Pressure value.
        """
        return self._pressure[port]

    def ramp(self, amplitude=-230., duration=1.5, port=0):
        """
        Creates a pressure ramp over the specified duration.

        Args:
            amplitude (float, optional): Maximum pressure. Defaults to -230.
            duration (float, optional): Ramp duration in seconds. Defaults to 1.5.
            port (int, optional): Pressure port number. Defaults to 0.
        """
        t0 = time.time()
        t = t0
        while t - t0 < duration:
            self.set_pressure(int(amplitude * (t - t0) / duration), port)
            t = time.time()
        self.set_pressure(0, port)

    def set_ATM(self, atm):
        """
        Sets the atmospheric pressure state.

        Args:
            atm (bool): True for atmospheric mode, False otherwise.
        """
        self.state = atm

    def get_ATM(self):
        """
        Gets the current atmospheric pressure state.

        Returns:
            bool: Atmospheric mode state.
        """
        return self.state

    # --- Pressure Acquisition System Methods ---
    def start_acquisition(self, interval=0.05, callback=None):
        """
        Starts continuous pressure acquisition in a separate thread.
        Child classes should call this method to begin measuring pressure.
        
        Args:
            interval (float, optional): Time between measurements (s). Defaults to 0.05.
            callback (callable, optional): Optional function called with each measurement.
        """
        self.info("Starting pressure acquisition.")
        if self._pressure_acq_thread is None:
            self._pressure_acq_thread = PressureAcquisitionThread(self, interval, callback)
            self._pressure_acq_thread.start()

    def get_last_acquisition(self):
        """
        Returns the most recent numeric pressure measurement.
        Returns None if no data is available.

        Returns:
            int or None: Last measurement, or None if no data available.
        """
        if self._pressure_acq_thread:
            return self._pressure_acq_thread.get_last_data()
        return None

    def get_last_acquisition_sample(self):
        """Return an atomic (measured pressure, acquisition time) pair."""
        thread = self._pressure_acq_thread
        if thread is not None and thread._last_data_queue:
            return thread._last_data_queue[-1]
        return None, None

    def stop_acquisition(self):
        """
        Stops the continuous pressure acquisition thread.
        """
        self.info("Stopping pressure acquisition.")
        if self._pressure_acq_thread:
            self._pressure_acq_thread.stop()
            self._pressure_acq_thread.join()
            self._pressure_acq_thread = None


class FakePressureController(PressureController):
    """
    Fake pressure controller for testing purposes.
    Continuously provides simulated pressure values.
    """
    def __init__(self):
        """
        Initializes the fake controller and starts automatic acquisition.
        """
        super(FakePressureController, self).__init__()
        self.pressure = 0
        # Start the acquisition thread automatically so that
        # ``get_last_acquisition`` returns values when used in the
        # fake rig setup.  Without this the GUI raises an exception
        # because no pressure measurements are available.
        self.start_acquisition()

    def measure(self, port=0):
        """
        Returns a fake pressure measurement.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.

        Returns:
            int: Simulated pressure value.
        """
        return self.pressure

    def set_pressure(self, pressure, port=0):
        """
        Sets a fake pressure value and logs the update.

        Args:
            pressure (int): Desired pressure.
            port (int, optional): Pressure port number. Defaults to 0.
        """
        self.debug('Pressure set to: {}'.format(pressure))
        self.pressure = pressure

    def get_pressure(self, port=0):
        """
        Returns the current simulated pressure.

        Args:
            port (int, optional): Pressure port number. Defaults to 0.

        Returns:
            int: Simulated pressure.
        """
        return self.pressure
