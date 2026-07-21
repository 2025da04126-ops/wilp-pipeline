"""
Pipeline Orchestration Module for RecoMart Pipeline.
Defines the end-to-end DAG: Ingestion -> Validation -> Preparation ->
Transformation -> Feature Store -> Model Training.
Uses Prefect for workflow management with logging and error handling.
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import BASE_DIR, LOGS_DIR
from src.logger import get_logger

logger = get_logger("orchestration")


# ============================================================================
# Pipeline Tasks (can be used standalone or with Prefect)
# ============================================================================

def task_generate_data() -> Dict:
    """Task 1: Generate synthetic data if not present."""
    logger.info("[TASK] Generating synthetic data...")
    from src.generate_data import save_data
    result = save_data()
    return {'status': 'success', 'datasets': len(result)}


def task_ingest_data() -> Dict:
    """Task 2: Ingest data from all sources."""
    logger.info("[TASK] Running data ingestion...")
    from src.ingestion.ingest_data import ingest_all_sources
    result = ingest_all_sources()
    return result


def task_validate_data() -> Dict:
    """Task 3: Validate data quality."""
    logger.info("[TASK] Running data validation...")
    from src.validation.validate_data import validate_all_sources
    result = validate_all_sources()
    return result


def task_prepare_data() -> Dict:
    """Task 4: Clean and prepare data."""
    logger.info("[TASK] Running data preparation...")
    from src.preparation.prepare_data import prepare_all_data
    result = prepare_all_data()
    return result


def task_transform_data() -> Dict:
    """Task 5: Engineer features."""
    logger.info("[TASK] Running feature engineering...")
    from src.transformation.transform_data import transform_all_data
    result = transform_all_data()
    return result


def task_feature_store() -> Dict:
    """Task 6: Populate feature store."""
    logger.info("[TASK] Running feature store pipeline...")
    from src.feature_store.feature_store import run_feature_store
    result = run_feature_store()
    return result


def task_train_model() -> Dict:
    """Task 7: Train and evaluate models."""
    logger.info("[TASK] Running model training...")
    from src.training.train_model import train_and_evaluate
    result = train_and_evaluate()
    return result


# ============================================================================
# Pipeline DAG Definition
# ============================================================================

class PipelineDAG:
    """
    End-to-end pipeline DAG orchestrator.
    Executes tasks in sequence with error handling, logging, and monitoring.
    """
    
    def __init__(self):
        self.tasks = [
            ('generate_data', task_generate_data),
            ('ingest_data', task_ingest_data),
            ('validate_data', task_validate_data),
            ('prepare_data', task_prepare_data),
            ('transform_data', task_transform_data),
            ('feature_store', task_feature_store),
            ('train_model', task_train_model),
        ]
        self.results = {}
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def run(self, start_from: int = 0, stop_at: int = None) -> Dict:
        """
        Execute the pipeline DAG.
        
        Args:
            start_from: Task index to start from (0-based)
            stop_at: Task index to stop at (exclusive)
        """
        logger.info("=" * 70)
        logger.info(f"RECOMART PIPELINE - Run ID: {self.run_id}")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        pipeline_start = time.time()
        tasks_to_run = self.tasks[start_from:stop_at]
        
        overall_status = 'success'
        
        for i, (task_name, task_func) in enumerate(tasks_to_run):
            step_num = start_from + i + 1
            total_steps = len(self.tasks)
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Step {step_num}/{total_steps}: {task_name.upper()}")
            logger.info(f"{'=' * 60}")
            
            task_start = time.time()
            
            try:
                result = task_func()
                task_duration = round(time.time() - task_start, 2)
                
                self.results[task_name] = {
                    'status': 'success',
                    'duration_sec': task_duration,
                    'result': result if isinstance(result, dict) else str(result)
                }
                
                logger.info(f"[OK] {task_name} completed in {task_duration}s")
                
            except Exception as e:
                task_duration = round(time.time() - task_start, 2)
                
                self.results[task_name] = {
                    'status': 'failed',
                    'duration_sec': task_duration,
                    'error': str(e)
                }
                
                logger.error(f"[FAIL] {task_name} FAILED after {task_duration}s: {str(e)}")
                overall_status = 'partial_failure'
                
                # Continue with next task (graceful degradation)
                logger.warning(f"Continuing pipeline despite failure in {task_name}")
        
        pipeline_duration = round(time.time() - pipeline_start, 2)
        
        # Generate pipeline run report
        run_report = {
            'run_id': self.run_id,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration_sec': pipeline_duration,
            'overall_status': overall_status,
            'tasks_executed': len(tasks_to_run),
            'tasks_succeeded': len([r for r in self.results.values() if r['status'] == 'success']),
            'tasks_failed': len([r for r in self.results.values() if r['status'] == 'failed']),
            'task_results': self.results
        }
        
        # Save run report
        report_path = os.path.join(LOGS_DIR, f'pipeline_run_{self.run_id}.json')
        with open(report_path, 'w') as f:
            json.dump(run_report, f, indent=2, default=str)
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"PIPELINE COMPLETE - Status: {overall_status}")
        logger.info(f"Duration: {pipeline_duration}s")
        logger.info(f"Tasks: {run_report['tasks_succeeded']}/{run_report['tasks_executed']} succeeded")
        logger.info(f"Report: {report_path}")
        logger.info(f"{'=' * 70}")
        
        return run_report


# ============================================================================
# Prefect Flow Definition (optional - runs if Prefect is available)
# ============================================================================

def create_prefect_flow():
    """Create a Prefect flow for pipeline orchestration (if Prefect is installed)."""
    try:
        from prefect import flow, task
        
        @task(name="generate_data", retries=2, retry_delay_seconds=5)
        def prefect_generate_data():
            return task_generate_data()
        
        @task(name="ingest_data", retries=2, retry_delay_seconds=5)
        def prefect_ingest_data():
            return task_ingest_data()
        
        @task(name="validate_data", retries=1)
        def prefect_validate_data():
            return task_validate_data()
        
        @task(name="prepare_data", retries=1)
        def prefect_prepare_data():
            return task_prepare_data()
        
        @task(name="transform_data", retries=1)
        def prefect_transform_data():
            return task_transform_data()
        
        @task(name="feature_store", retries=1)
        def prefect_feature_store():
            return task_feature_store()
        
        @task(name="train_model", retries=1)
        def prefect_train_model():
            return task_train_model()
        
        @flow(name="recomart_pipeline")
        def recomart_pipeline_flow():
            """RecoMart End-to-End Data Management Pipeline."""
            data = prefect_generate_data()
            ingestion = prefect_ingest_data(wait_for=[data])
            validation = prefect_validate_data(wait_for=[ingestion])
            preparation = prefect_prepare_data(wait_for=[validation])
            transformation = prefect_transform_data(wait_for=[preparation])
            features = prefect_feature_store(wait_for=[transformation])
            model = prefect_train_model(wait_for=[features])
            return model
        
        return recomart_pipeline_flow
        
    except ImportError:
        logger.warning("Prefect not available. Using standalone DAG runner.")
        return None


# ============================================================================
# Main Execution
# ============================================================================

def run_pipeline(use_prefect: bool = False) -> Dict:
    """
    Run the complete RecoMart pipeline.
    
    Args:
        use_prefect: Whether to use Prefect for orchestration
    """
    if use_prefect:
        flow = create_prefect_flow()
        if flow is not None:
            return flow()
    
    # Use standalone DAG runner
    dag = PipelineDAG()
    return dag.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RecoMart Pipeline Orchestrator')
    parser.add_argument('--prefect', action='store_true', help='Use Prefect for orchestration')
    parser.add_argument('--start', type=int, default=0, help='Start from task index')
    parser.add_argument('--stop', type=int, default=None, help='Stop at task index')
    
    args = parser.parse_args()
    
    if args.prefect:
        result = run_pipeline(use_prefect=True)
    else:
        dag = PipelineDAG()
        result = dag.run(start_from=args.start, stop_at=args.stop)
    
    print(f"\nPipeline finished with status: {result.get('overall_status', 'unknown')}")
