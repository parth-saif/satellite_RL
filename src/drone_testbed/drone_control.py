""""
DroneController class:
1. Log data from drones.
2. Edit parameters of drones.
3. Upload pre-computed trajectories to drones.

TODO: Implement RPOController child class and add method to run the experiment.#

"""

import sys
import time
import numpy as np

from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.utils.reset_estimator import reset_estimator
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

class DroneController(): # base drone controller class
    def __init__(self, uri_set):
        self.uri_set = uri_set
        self.uri_list = list(self.uri_set)
        self.swarm_size = len(self.uri_list)
        self.state_logs = [[] for _ in range(self.swarm_size)]

    def activate_hl_commander(self, scf): # set high level commander to on
        scf.cf.param.set_value('commander.enHighLevel', '1')
    
    def activate_mellinger(self, scf): # activate Mellinger PID-like controller
        scf.cf.param.set_value('stabilizer.controller', '2') 

    def upload_trajectory(self, cf, trajectory_id, trajectory): # upload a trajectory into 1 crazyflie 
        trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]

        trajectory_mem.trajectory = trajectory

        upload_result = trajectory_mem.write_data_sync()
        if not upload_result:
            print('Upload failed, aborting!')
            sys.exit(1)
            cf.high_level_commander.define_trajectory(
            trajectory_id,
            0,
            len(trajectory),
            type=HighLevelCommander.TRAJECTORY_TYPE_POLY4D_COMPRESSED)

        total_duration = 0
        # Skip the start element
        for segment in trajectory[1:]:
            total_duration += segment.duration

        return total_duration

    def arm(self, scf): # arm drones
        self.scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

    def state_callback(self, uri, data): # callback for logging state estimates
        num = self.uri_list.index(uri) # extract drone number
        self.state_logs[num].append((data['stateEstimate.x'],  data['stateEstimate.y'],  data['stateEstimate.z'],  
                                     data['stateEstimate.vx'],  data['stateEstimate.vy'],  data['stateEstimate.vz'],
                                     data['stateEstimate.yaw']
        )) # append to logs

    def start_position_printing(self, scf):
        log_conf1 = LogConfig(name='State', period_in_ms=500)
        log_conf1.add_variable('stateEstimate.x', 'float')
        log_conf1.add_variable('stateEstimate.y', 'float')
        log_conf1.add_variable('stateEstimate.z', 'float')
        log_conf1.add_variable('stateEstimate.vx', 'float')
        log_conf1.add_variable('stateEstimate.vy', 'float')
        log_conf1.add_variable('stateEstimate.vz', 'float')
        log_conf1.add_variable('stateEstimate.yaw', 'float')

        scf.cf.log.add_config(log_conf1)
        log_conf1.data_received_cb.add_callback(lambda _timestamp, data, _logconf: self.state_callback(scf.cf.link_uri, data))
        log_conf1.start()

# RPO controller child class
class RPOController(DroneController): # inherit from base drone controller
    def __init__(self, uri_list, ):
        super().__init__(uri_list)

if __name__ == '__main__':
    URI1 = 'radio://0/80/2M/E7E7E7E7E7'
    URI2 = 'radio://0/80/2M/E7E7E7E7E8'

    uri_set = { # for some reason, swarm code for crazyflie uses sets instead of lists, probably for backend parallel.
    URI1,
    URI2,}

    controller = DroneController(uri_set)

    # # Test callback function
    # print(controller.uri_list)
    # import random
    # num_records = 2
    # for _ in range(num_records):
    #     data = {
    #         'stateEstimate.x': round(random.uniform(-100.0, 100.0), 2),  # Position X (e.g., meters)
    #         'stateEstimate.y': round(random.uniform(-100.0, 100.0), 2),  # Position Y
    #         'stateEstimate.z': round(random.uniform(0.0, 50.0), 2),    # Position Z (e.g., altitude)
    #         'stateEstimate.vx': round(random.uniform(-5.0, 5.0), 2),   # Velocity X (e.g., m/s)
    #         'stateEstimate.vy': round(random.uniform(-5.0, 5.0), 2),   # Velocity Y
    #         'stateEstimate.vz': round(random.uniform(-1.0, 1.0), 2),   # Velocity Z (e.g., climb/descent rate)
    #         'stateEstimate.yaw': round(random.uniform(-180.0, 180.0), 2) # Yaw (e.g., degrees)
    #     }
    #     controller.state_callback(controller.uri_list[0], data)
    #     controller.state_callback(controller.uri_list[1], data)
    # print(controller.state_logs)

    # cflib.crtp.init_drivers()

    # factory = CachedCfFactory(rw_cache='./cache')

    # with Swarm(uris, factory=factory) as swarm:
    #     swarm.reset_estimators()

    #     swarm.parallel(controller.start_position_printing)
