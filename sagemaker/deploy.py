import sagemaker
from sagemaker.sklearn import SKLearn

role = "arn:aws:iam::<your-account-id>:role/<your-sagemaker-role>"
bucket = "your-s3-bucket-name"

sklearn_estimator = SKLearn(
    entry_point='src/train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    s3_output_path=f's3://{bucket}/output'
)

# Start training
sklearn_estimator.fit({'train': f's3://{bucket}/data/sample.csv'})

# Deploy model
predictor = sklearn_estimator.deploy(
    instance_type='ml.m5.large',
    endpoint_name='ml-endpoint'
)
