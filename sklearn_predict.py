import os
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data

import joblib
import pandas as pd

# config
# project base
BASE_PATH = r"C:\Users\asus\Desktop\testtesttest"
# dataset file & model file
DATA = 'breast_homo_test.csv'
MODEL = '2022061814380276428431_model.pkl'
# server ip & port
FLOW_IP = "127.0.0.1"
FLOW_PORT = "9380"

# pipeline

os.system(f"pipeline init --ip {FLOW_IP} --port {FLOW_PORT}")

reader_1 = Reader(name="reader_1")
reader_1.get_party_instance(role="guest", party_id=9999).component_param(table={"name": "breast_homo_guest", "namespace": "experiment"})
reader_1.get_party_instance(role="host", party_id=10000).component_param(table={"name": "breast_homo_host", "namespace": "experiment"})

evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

pipeline = PipeLine.load_model_from_file(MODEL)
pipeline.deploy_component([pipeline.data_transform_0, pipeline.intersect_0, pipeline.homo_lr_0])

predict_pipeline = PipeLine()

predict_pipeline.add_component(reader_1)\
                .add_component(pipeline, data=Data(predict_input={pipeline.data_transform_0.input.data: reader_1.output.data}))\
                .add_component(evaluation_0, data=Data(data=pipeline.homo_lr_0.output.data))

predict_pipeline.compile()
predict_pipeline.predict()

print(json.dumps(pipeline.get_component("homo_lr_0").get_summary(), indent=4, ensure_ascii=False))
print(json.dumps(pipeline.get_component("evaluation_0").get_summary(), indent=4, ensure_ascii=False))

# sklearn
'''
testdata = pd.read_csv(os.path.join(BASE_PATH, DATA))

lr = joblib.load(os.path.join(BASE_PATH, MODEL))

print(lr.predict(testdata))
'''