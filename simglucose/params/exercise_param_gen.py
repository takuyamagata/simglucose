import pandas as pd
import pkg_resources

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

para = pd.read_csv(PATIENT_PARA_FILE)
