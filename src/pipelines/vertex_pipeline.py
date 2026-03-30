import argparse
import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

def create_vertex_pipeline(config_path='configs/project_config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config['gcp']['project_id'],
        location=config['gcp']['region'],
        staging_bucket=config['gcp']['staging_bucket']
    )
    
    # Define pipeline
    @aiplatform.pipeline
    def churn_pipeline():
        # Pipeline components
        pass
    
    return churn_pipeline

def run_pipeline(dry_run=False):
    if dry_run:
        print("🔍 Dry run - pipeline would be submitted to Vertex AI")
        return
    
    # Submit pipeline job
    pipeline = create_vertex_pipeline()
    job = pipeline_jobs.PipelineJob(
        display_name="churn-prediction-pipeline",
        template_path="pipeline_spec.yaml",
        parameter_values={}
    )
    job.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    run_pipeline(dry_run=args.dry_run)