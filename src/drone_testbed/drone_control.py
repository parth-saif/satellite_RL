""""
DroneController class:
1. Log data from drones.
2. Edit parameters of drones.
3. Upload pre-computed trajectories to drones.
---
RPOController:
Inherits from base DroneController, specifically for RPO experiments
Adds the following functionalities:
- upload orbits and rendezvous trajectories to relevant drones.
- 

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
    """
    Base Drone controller class.
    """
    def __init__(self, uri_set: set):
        """
        Initialise swarm.
        :param uri_set: Set of URIs for drones in the swarm.
        """
        self.uri_set = uri_set
        self.uri_list = list(self.uri_set)
        self.swarm_size = len(self.uri_list)
        self.state_logs = [[] for _ in range(self.swarm_size)]

    def activate_hl_commander(self, scf): # set high level commander to ON
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
    """
    Controller for RPO missions with drone swarm.
    """
    def __init__(self, uri_list, trajectories):
        super().__init__(uri_list)
        self.target_sat_uri = self.uri_list[0] # assume target sat is at index 0
        self.chaser_sat_uri = self.uri_list[1] # chaser defined as second

        self.target_orb, self.chaser_orb, self.chaser_rendez = trajectories
    
    def upload_trajectories(self, scf): # uploading rendezvous manoeuvre
        """
        Upload rendezvous trajectory to chaser drone only.
        :param scf: Swarm drone instance
        :param traj_id: Trajectory id for rendezvous manoeuvre
        :param trajectory: Individual drone's trajectory
        """
        if scf.__dict__['_link_uri'] == self.target_sat_uri: # only upload to target
            # upload orbit
            traj_id = 1
            self.upload_trajectory(scf.cf, traj_id, self.target_orb)
        elif scf.__dict__['_link_uri'] == self.chaser_sat_uri:
            # upload orbit
            traj_id = 1
            self.upload_trajectory(scf.cf, traj_id, self.chaser_orb)

            # upload manoeuvre
            traj_id = 2
            self.upload_trajectory(scf.cf, traj_id, self.chaser_rendez)

            # upload tracking -> same as trajectory of target
            traj_id = 3
            self.upload_trajectory(scf.cf, traj_id, self.target_orb)

    def run_sequence(self,scf, **kwargs):
        """
        Run an RPO sequence demonstration. 
        1. Arm and take-off.
        2. Begin orbits.
        3. After some time, initiate rendezvous manoeuvre.
        4. Land both drones and disarm.

        :param scf: Swarm drone instance
        :param **kwargs: variables to configure the sequence.
        """
        # Arm the Crazyflie
        self.arm(scf)

        # activate HL commander
        self.activate_hl_commander(scf)

        #set mellinger PID controller
        self.activate_mellinger(scf)
        commander = scf.cf.high_level_commander

        # set group masks for each drone
        if scf.__dict__['_link_uri'] == self.chaser_sat_uri:
            commander.set_group_mask(1) # chaser is group 1
        elif scf.__dict__['_link_uri'] == self.target_sat_uri:
            commander.set_group_mask(2) # target is group 2

        commander.takeoff(kwargs['take_off_h'], kwargs['take_off_dur']) # take off
        time.sleep(3.0)

        rendezvous_time = kwargs['rendez_time']
        for i in range(kwargs['num_orbits']): # run orbits
            commander.start_trajectory(trajectory_id = 1, time_scale=1.0, relative_position = True, group_mask=0) # orbit, group mask = 0 means applied to all drones
            start_time = time.time()
            end_time = start_time + rendezvous_time
            if time.time() > end_time: # at rendezvous time, tell chaser to do manoeuvre
                commander.start_trajectory(trajectory_id=2, group_mask=1)
                # might need a sleep here?
                commander.start_trajectory(trajector_id=3, group_mask=1)
                break # do one orbit and break?
        
        # land
        time.sleep(0.2)
        commander.land(0.0, kwargs['land_dur'])
        commander.stop() # disarm

if __name__ == '__main__':
    URI1 = 'radio://0/80/2M/E7E7E7E7E7'
    URI2 = 'radio://0/80/2M/E7E7E7E7E8'

    # for some reason, swarm code for crazyflie uses sets instead of lists, probably for backend parallel.
    uri_set = { 
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

    # with Swarm(uri_set, factory=factory) as swarm:
    #     swarm.reset_estimators()

    #     swarm.parallel(controller.start_position_printing)
