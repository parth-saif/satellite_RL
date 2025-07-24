import time
import logging

# crazyflie api imports
import cflib.crtp # -> scanning for available Crazyflies
from cflib.crazyflie import Crazyflie # -> Crazyflie class tp connect/send/recieve data
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie # -> synchronous interface to Crazyflie (turns into blocking calls)

from cflib.crazyflie.log import LogConfig # -> configuration for logging data from Crazyflie
from cflib.crazyflie.syncLogger import SyncLogger # -> synchronous logger to log data from Crazyflie
from cflib.utils import uri_helper # -> parse URIs for Crazyflie connections

# define uri for Crazyflie connection
uri = 'radio://0/80/2M/E7E7E7E7E7'
logging.basicConfig(level=logging.ERROR) # Set logging level to ERROR to reduce output noise

def connect_crazyflie():
    print("Connected to Crazyflie at URI:", uri)
    time.sleep(3)
    print("Disconnecting from Crazyflie at URI:", uri)

def log_async_data(scf, logconf):
    cf = scf.cf  # Get the Crazyflie instance from the SyncCrazyflie instance
    cf.log.add_config(logconf)  # Add the log configuration to the Crazyflie instance
    logconf.data_received_cb.add_callback(log_stab_callback)  # Add the callback function to handle received data asynchronously
    logconf.start()
    time.sleep(5)
    logconf.stop()
    
def log_stab_callback(timestamp, data, logconf): # callback function for logging data 
    print('[%d][%s]: %s' % (timestamp, logconf.name, data))


def log_data(scf, logconf):
# scf is the SyncCrazyflie instance, logconf is the LogConfig instance
    with SyncLogger(scf, lg_stab) as logger: # Synchronous logger to log data from Crazyflie 

            for log_entry in logger:

                timestamp = log_entry[0]
                data = log_entry[1]
                logconf_name = log_entry[2]

                print('[%d][%s]: %s' % (timestamp, logconf_name, data))

                break

def param_stab_est_callback(name, value): # callback function for parameter editing
    print("Parameter '%s' changed to '%s'" % (name, value))

def param_async(scf, groupstr, namestr): # asynchronous parameter editing
    # input string group and name of the parameter to edit
    cf = scf.cf 
    full_param_name = groupstr+"."+namestr # full parameter name
    cf.param.add_update_callback(group = groupstr, name = namestr, cb = param_stab_est_callback) # add callback function to parameter
    time.sleep(1) # wait for response and not lose connection immediately

    cf.param.set_value(full_param_name, 2) # set the parameter value to 2
    time.sleep(1) # wait for the parameter to be set
    cf.param.set_value(full_param_name, 1) # set the parameter value back to 0
    time.sleep(1) # wait for the parameter to be set back

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # choose variables to log
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10) # LogConfig for stabilizer data
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')

    #edit parameters
    group = 'stabilizer' # group of parameters to edit
    name = 'estimator' # name of the parameter to edit e.g state estimator used

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        # check if the Crazyflie is connected
        #connect_crazyflie()
        # Log data from Crazyflie
        #log_data(scf, lg_stab)

        # Log data asynchronously
        log_async_data(scf, lg_stab)

        #asynchronous parameter editing
        param_async(scf, group, name)