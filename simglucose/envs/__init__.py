from gym.envs.registration import register

register(
    id='simGlucose-v0',
    entry_point='simglucose.envs.simglucose_gym_env:T1DSimEnv'
    )
