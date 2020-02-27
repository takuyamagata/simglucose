from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatientApprox import T1DPatient
# from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import pandas as pd
import numpy as np
import pkg_resources
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from datetime import datetime

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, patient_name=None, reward_fun=None, observe_internal_state=False, squash_action=False):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        seeds = self.seed()
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = 'adolescent#001'
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('GuardianRT', seed=seeds[1])
        hour = self.np_random.randint(low=0.0, high=24.0)
        # start_time = datetime(2018, 1, 1, hour, 0, 0) # random start time
        start_time = datetime(2018, 1, 1, 6, 0, 0) # fixed start time (6am)
        scenario = RandomScenario(start_time=start_time, seed=seeds[2])
        pump = InsulinPump.withName('Insulet')
        self.observe_internal_state = observe_internal_state
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)
        self.reward_fun = reward_fun
        self.squash_action = squash_action

    @staticmethod
    def pick_patient():
        # TODO: cannot be used to pick patient at the env constructing space
        # for now
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        while True:
            print('Select patient:')
            for j in range(len(patient_params)):
                print('[{0}] {1}'.format(j + 1, patient_params['Name'][j]))
            try:
                select = int(input('>>> '))
            except ValueError:
                print('Please input a number.')
                continue

            if select < 1 or select > len(patient_params):
                print('Please input 1 to {}'.format(len(patient_params)))
                continue

            return select

    def step(self, action):
        # This gym only controls basal insulin
        if self.squash_action:
            basal_rate = 0.0136
            if action < 0:
                basal_ = np.exp(action)
            else:
                basal_ = 5.0 - (5.0 - 1.0)*np.exp(-1.0/(5.0-1.0)*action)
            basal_ = basal_ * basal_rate
        else:
            basal_ = action
        
        act = Action(basal=basal_, bolus=0, meal=0)
        if self.reward_fun is None:
            step = self.env.step(act)
        else:
            step = self.env.step(act, reward_fun=self.reward_fun)
        
        if self.observe_internal_state:
            # observation includes all internal states
            observation = np.append(step.observation.CGM, step.info['patient_state'])
        else:
            observation = np.array(step.observation.CGM)
        return observation, step.reward, step.done, step.info

    def reset(self):
        # new scenario
        seeds = self.seed()
        hour = self.np_random.randint(low=0.0, high=24.0)
        # start_time = datetime(2018, 1, 1, hour, 0, 0) # random start time
        start_time = datetime(2018, 1, 1, 6, 0, 0) # fixed start time (6am)
        self.env.scenario = RandomScenario(start_time=start_time, seed=seeds[2])
        
        obs, _, _, info = self.env.reset()
        if self.observe_internal_state:
            observation = np.append(obs.CGM, info['patient_state'])
        else:
            observation = np.array(obs.CGM)
        return observation

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=-np.inf, high=+np.inf, shape=(1,))
        #return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        if self.observe_internal_state:
            return spaces.Box(low=0, high=np.inf, shape=(14,))
        else:
            return spaces.Box(low=0, high=np.inf, shape=(1,))
        