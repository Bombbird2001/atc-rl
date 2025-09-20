import platform

os_name = platform.system()
if os_name == "Windows":
    import mmap
    import win32event
    import struct
elif os_name == "Darwin" or os_name == "Linux":
    import posix_ipc


from abc import ABC, abstractmethod


class GameBridge(ABC):
    @abstractmethod
    def signal_reset_sim(self):
        raise NotImplementedError

    @abstractmethod
    def wait_action_ready(self):
        raise NotImplementedError

    @abstractmethod
    def signal_action_done(self):
        raise NotImplementedError

    @abstractmethod
    def signal_reset_after_step(self):
        raise NotImplementedError

    @abstractmethod
    def get_total_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_aircraft_state(self):
        raise NotImplementedError

    @abstractmethod
    def write_actions(self, hdg_action, alt_action, spd_action):
        raise NotImplementedError

    @classmethod
    def get_bridge_for_platform(cls, **kwargs):
        if os_name == "Windows":
            return WindowsGameBridge(**kwargs)
        elif os_name == "Darwin" or os_name == "Linux":
            return UnixGameBridge(**kwargs)
        else:
            raise NotImplementedError(f"Unknown OS {os_name}")


class WindowsGameBridge(GameBridge):
    # 52 bytes shared region: [proceed flag(1 byte)] [action(3 bytes)] [reward(4 bytes (1 float))] [terminated(1 byte)] [empty padding(2 bytes)] [state(40 bytes (7x floats, 3x ints))]
    FILE_SIZE = 52
    STRUCT_FORMAT = "bbbbf?xxxfffffffiii"

    def __init__(self, instance_suffix=""):
        # Create anonymous memory-mapped file with a local name
        self.mm = mmap.mmap(-1, self.__class__.FILE_SIZE, tagname=f"Local\\ATCRLSharedMem{instance_suffix}")

        # Named events for synchronization
        self.reset_sim = win32event.CreateEvent(None, False, False, f"Local\\ATCRLResetEvent{instance_suffix}")
        self.action_ready = win32event.CreateEvent(None, False, False, f"Local\\ATCRLActionReadyEvent{instance_suffix}")
        self.action_done = win32event.CreateEvent(None, False, False, f"Local\\ATCRLActionDoneEvent{instance_suffix}")
        self.reset_after_step = win32event.CreateEvent(None, False, False, f"Local\\ATCRLResetAfterEvent{instance_suffix}")

    def signal_reset_sim(self):
        win32event.SetEvent(self.reset_sim)

    def wait_action_ready(self):
        win32event.WaitForSingleObject(self.action_ready, win32event.INFINITE)

    def signal_action_done(self):
        win32event.SetEvent(self.action_done)

    def signal_reset_after_step(self):
        win32event.SetEvent(self.reset_after_step)

    def get_total_state(self) -> tuple:
        self.mm.seek(0)
        return struct.unpack(self.__class__.STRUCT_FORMAT, self.mm.read(self.__class__.FILE_SIZE))

    def get_aircraft_state(self) -> tuple:
        self.mm.seek(12)
        return struct.unpack(self.__class__.STRUCT_FORMAT[9:], self.mm.read(self.__class__.FILE_SIZE - 12))

    def write_actions(self, hdg_action, alt_action, spd_action):
        self.mm.seek(0)
        self.mm.write(struct.pack("bbbb", 1, hdg_action, alt_action, spd_action))


class UnixGameBridge(GameBridge):
    # 52 bytes shared region: [proceed flag(1 byte)] [action(3 bytes)] [reward(4 bytes (1 float))] [terminated(1 byte)] [empty padding(2 bytes)] [state(40 bytes (7x floats, 3x ints))]
    FILE_SIZE = 52
    STRUCT_FORMAT = "bbbbf?xxxfffffffiii"

    def __init__(self, instance_suffix=""):
        shm = posix_ipc.SharedMemory(f"ATCRLSharedMem{instance_suffix}", posix_ipc.O_CREX, size=self.__class__.FILE_SIZE)
        self.mm = mmap.mmap(shm.fd, shm.size)
        shm.close_fd()

        self.reset_sim = posix_ipc.Semaphore(f"ATCRLResetEvent{instance_suffix}")
        self.action_ready = posix_ipc.Semaphore(f"ATCRLActionReadyEvent{instance_suffix}")
        self.action_done = posix_ipc.Semaphore(f"ATCRLActionDoneEvent{instance_suffix}")
        self.reset_after_step = posix_ipc.Semaphore(f"ATCRLResetAfterEvent{instance_suffix}")

    def signal_reset_sim(self):
        self.reset_sim.release()

    def wait_action_ready(self):
        self.action_ready.acquire()

    def signal_action_done(self):
        self.action_done.release()

    def signal_reset_after_step(self):
        self.reset_after_step.release()

    def get_total_state(self) -> tuple:
        self.mm.seek(0)
        return struct.unpack(self.__class__.STRUCT_FORMAT, self.mm.read(self.__class__.FILE_SIZE))

    def get_aircraft_state(self) -> tuple:
        self.mm.seek(12)
        return struct.unpack(self.__class__.STRUCT_FORMAT[9:], self.mm.read(self.__class__.FILE_SIZE - 12))

    def write_actions(self, hdg_action, alt_action, spd_action):
        self.mm.seek(0)
        self.mm.write(struct.pack("bbbb", 1, hdg_action, alt_action, spd_action))
