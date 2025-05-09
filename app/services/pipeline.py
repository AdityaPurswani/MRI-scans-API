import shutil
import os
import sys
import nibabel as nib
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
import tempfile
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any
import requests 
import subprocess # Kept for DummyMRIScansPipeline if it were to simulate calls

# --- Load environment variables ---
try:
    # This path assumes this script (app/pipeline.py) is one level down from 'app', 
    # and .env is in the project root.
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"pipeline.py: .env loaded from: {dotenv_path}")
    else:
        logging.warning(f"pipeline.py: .env file not found at expected path: {dotenv_path}. Relying on environment variables being set externally.")
except Exception as e:
    logging.error(f"pipeline.py: Error loading .env file: {e}")

# --- Import your actual processing classes ---
# These are expected to be in a file like MRIScansPipeline.py at the project root.
MRIScansPipeline_class, ImageSegmenter_class, FeatureExtraction_class = None, None, None
try:
    # This assumes your project root (containing MRIScansPipeline.py) is in PYTHONPATH.
    from MRIScansPipeline import MRIScansPipeline as ActualMRIScansPipeline
    from MRIScansPipeline import ImageSegmenter as ActualImageSegmenter
    from MRIScansPipeline import FeatureExtraction as ActualFeatureExtraction
    MRIScansPipeline_class = ActualMRIScansPipeline
    ImageSegmenter_class = ActualImageSegmenter
    FeatureExtraction_class = ActualFeatureExtraction
    logging.info("Successfully imported actual MRIScansPipeline, ImageSegmenter, and FeatureExtraction classes from MRIScansPipeline.py.")
except ImportError as e:
    logging.error(
        f"Could not import actual MRIScansPipeline, ImageSegmenter, FeatureExtraction classes from MRIScansPipeline.py. Error: {e}. "
        "Ensure MRIScansPipeline.py is in the project root and accessible. Falling back to DUMMY classes defined in this file."
    )
    # Fallback DUMMY classes 
    class DummyMRIScansPipeline:
        def __init__(self, input_dir, output_dir): # MODIFIED: Removed bet_executable_path
            self.input = input_dir
            self.output = output_dir
            os.makedirs(self.output, exist_ok=True)
            logging.warning(f"Using DUMMY MRIScansPipeline: input_dir={input_dir}, output_dir={output_dir}")
        
        def extract_brain_part(self, input_image_path, output_path_filename):
            full_output_path = os.path.join(self.output, output_path_filename)
            logging.info(f"DUMMY MRIScansPipeline: Simulating brain extraction from {input_image_path} to {full_output_path} (expecting fsl.wrappers.bet in actual class)")
            if os.path.exists(input_image_path): shutil.copy(input_image_path, full_output_path)
            else: logging.error(f"DUMMY MRIScansPipeline: Input file {input_image_path} not found for brain extraction.")
            return full_output_path
        
        def intensity_normalisation(self, brain_data):
            logging.info("DUMMY MRIScansPipeline: Normalizing intensity")
            if brain_data is not None and hasattr(brain_data, 'min') and hasattr(brain_data, 'max') and hasattr(brain_data, 'size') and brain_data.size > 0:
                min_val = brain_data.min()
                max_val = brain_data.max()
                if max_val == min_val:
                    return brain_data - min_val 
                return (brain_data - min_val) / (max_val - min_val)
            return brain_data 
        
        def fsl_bias_correction(self, input_image_path, output_path):
            logging.info(f"DUMMY MRIScansPipeline: Simulating FSL bias correction from {input_image_path} to {output_path}")
            if os.path.exists(input_image_path): shutil.copy(input_image_path, output_path)
            else: logging.error(f"DUMMY MRIScansPipeline: Input file {input_image_path} not found for bias correction.")
            return output_path
        
        def image_registration_mni(self, moving_image_path, template_mni_path, output_prefix_dir):
            logging.info(f"DUMMY MRIScansPipeline: Simulating registration of {moving_image_path} to MNI template {template_mni_path}, output prefix dir {output_prefix_dir}")
            output_filename = "registered.nii.gz" 
            os.makedirs(output_prefix_dir, exist_ok=True)
            destination_path = os.path.join(output_prefix_dir, output_filename)
            if os.path.exists(moving_image_path):
                shutil.copy(moving_image_path, destination_path)
                logging.info(f"DUMMY MRIScansPipeline: Registered file 'created' at {destination_path}")
                return destination_path
            else:
                logging.error(f"DUMMY MRIScansPipeline: Input file {moving_image_path} not found for registration.")
                return None

    class DummyImageSegmenter:
        def __init__(self, images_dir, atlas_paths, output_dir): # MODIFIED: Removed bet_executable_path
            self.images_dir = images_dir
            self.atlas_paths = atlas_paths 
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            logging.warning(f"Using DUMMY ImageSegmenter: images_dir={images_dir}, output_dir={output_dir}")
            for atlas_name in self.atlas_paths.keys():
                os.makedirs(os.path.join(self.output_dir, f"segmented_{atlas_name}"), exist_ok=True)
        
        def process_single_image(self, image_info_tuple):
            filename, _ = image_info_tuple
            input_file_path = os.path.join(self.images_dir, filename) 
            logging.info(f"DUMMY ImageSegmenter: Simulating segmentation of {input_file_path}")
            if os.path.exists(input_file_path):
                for atlas_name in self.atlas_paths.keys():
                    base_filename_no_ext = filename.replace(".nii.gz", "")
                    output_subdir_for_atlas = os.path.join(self.output_dir, f"segmented_{atlas_name}")
                    dummy_output_path = os.path.join(output_subdir_for_atlas, f"{base_filename_no_ext}_segmented_{atlas_name}.nii.gz")
                    with open(dummy_output_path, 'w') as f: f.write("dummy segmented content") 
                    logging.info(f"DUMMY ImageSegmenter: Created dummy segmented file {dummy_output_path}")
            else:
                logging.error(f"DUMMY ImageSegmenter: Input file {input_file_path} not found for segmentation.")
            return f"DUMMY segmented {filename}" 

    class DummyFeatureExtraction:
        def __init__(self, images_dir, atlas_paths, output_dir): # MODIFIED: Removed bet_executable_path
            self.images_dir = images_dir
            self.atlas_paths = atlas_paths
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            logging.warning(f"Using DUMMY FeatureExtraction: images_dir={images_dir}, output_dir={output_dir}")
        
        def extract_features(self, segmented_atlas_dir_path, output_csv_filename): 
            output_csv_full_path = os.path.join(self.output_dir, output_csv_filename)
            logging.info(f"DUMMY FeatureExtraction: Simulating feature extraction from {segmented_atlas_dir_path} to {output_csv_full_path}")
            dummy_data = {"region1": 100.0, "region2": 150.5} 
            with open(output_csv_full_path, 'w') as f:
                f.write("region,volume\n")
                for region, volume in dummy_data.items():
                    f.write(f"{region},{volume}\n")
            logging.info(f"DUMMY FeatureExtraction: Created dummy CSV {output_csv_full_path}")
            return dummy_data 

    if MRIScansPipeline_class is None: MRIScansPipeline_class = DummyMRIScansPipeline
    if ImageSegmenter_class is None: ImageSegmenter_class = DummyImageSegmenter
    if FeatureExtraction_class is None: FeatureExtraction_class = DummyFeatureExtraction

# --- Configuration ---
DO_SPACES_REGION_NAME = os.getenv("DO_SPACES_REGION_NAME", "nyc3")
DO_SPACES_ENDPOINT_URL = os.getenv("DO_SPACES_ENDPOINT_URL")
DO_SPACES_ACCESS_KEY_ID = os.getenv("DO_SPACES_ACCESS_KEY_ID")
DO_SPACES_SECRET_ACCESS_KEY = os.getenv("DO_SPACES_SECRET_ACCESS_KEY")
DO_SPACES_BUCKET_NAME = os.getenv("DO_SPACES_BUCKET_NAME")

if not DO_SPACES_ENDPOINT_URL and DO_SPACES_REGION_NAME:
    DO_SPACES_ENDPOINT_URL = f"https://{DO_SPACES_REGION_NAME}.digitaloceanspaces.com"

if not logging.getLogger().handlers: 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRIProcessingPipelineWithSpaces:
    def __init__(self, input_object_key: str):
        self.input_object_key = input_object_key
        self.temp_local_dir = None 
        
        if not all([DO_SPACES_ACCESS_KEY_ID, DO_SPACES_SECRET_ACCESS_KEY, DO_SPACES_BUCKET_NAME, DO_SPACES_ENDPOINT_URL]):
            msg = "DigitalOcean Spaces configuration for user data is incomplete."
            logger.error(msg)
            self._cleanup_temp_dir() 
            raise ValueError(msg)
        
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=DO_SPACES_REGION_NAME,
                endpoint_url=DO_SPACES_ENDPOINT_URL,
                aws_access_key_id=DO_SPACES_ACCESS_KEY_ID,
                aws_secret_access_key=DO_SPACES_SECRET_ACCESS_KEY
            )
            logger.info("S3 client for user data initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client for user data: {e}.")
            self._cleanup_temp_dir()
            raise

        self.bucket_name = DO_SPACES_BUCKET_NAME

        input_basename_no_ext = os.path.splitext(os.path.splitext(os.path.basename(input_object_key))[0])[0]
        self.base_spaces_output_prefix = os.path.join("mri_pipeline_outputs", input_basename_no_ext)
        self.processed_spaces_prefix = os.path.join(self.base_spaces_output_prefix, "processed")
        self.uncompressed_spaces_prefix = os.path.join(self.base_spaces_output_prefix, "processed", "uncompressed")
        self.segmented_spaces_prefix = os.path.join(self.base_spaces_output_prefix, "processed", "segmented")
        self.features_spaces_prefix = os.path.join(self.base_spaces_output_prefix, "processed", "features")
        
        self.temp_local_dir = tempfile.mkdtemp(prefix="mri_processing_")
        logger.info(f"Created temporary local directory for processing: {self.temp_local_dir}")
        
        self.local_uploads_dir = os.path.join(self.temp_local_dir, "uploads")
        self.local_processed_dir = os.path.join(self.temp_local_dir, "processed")
        self.local_segmented_dir = os.path.join(self.local_processed_dir, "segmented")
        self.local_features_dir = os.path.join(self.local_processed_dir, "features")
        self.local_common_atlases_dir = os.path.join(self.temp_local_dir, "common_atlases")
        # self.local_tools_dir is no longer needed as BET script is not downloaded here.

        os.makedirs(self.local_uploads_dir, exist_ok=True)
        os.makedirs(self.local_processed_dir, exist_ok=True)
        os.makedirs(self.local_segmented_dir, exist_ok=True)
        os.makedirs(self.local_features_dir, exist_ok=True)
        os.makedirs(self.local_common_atlases_dir, exist_ok=True)
        # os.makedirs(self.local_tools_dir, exist_ok=True) # Not needed

        # --- BET Executable Handling Removed ---
        # The MRIScansPipeline_class is now expected to use fsl.wrappers.bet
        # and does not require a bet_executable_path.
        
        self.pipeline = MRIScansPipeline_class( 
            input_dir=self.local_uploads_dir,
            output_dir=self.local_processed_dir
            # MODIFIED: No longer passing bet_executable_path
        )
        logger.info(f"Initialized MRIScansPipeline (type: {type(self.pipeline).__name__}). Assumes FSL (including BET wrapper) is in environment PATH.")

    def _upload_to_spaces(self, local_file_path: str, object_key: str) -> Optional[str]:
        if not os.path.exists(local_file_path):
            logger.error(f"Local file not found for upload: {local_file_path}")
            return None
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, object_key)
            s3_uri = f"s3://{self.bucket_name}/{object_key}"
            logger.info(f"Successfully uploaded {local_file_path} to {s3_uri}")
            return s3_uri
        except FileNotFoundError: 
            logger.error(f"Local file disappeared before upload: {local_file_path}")
            return None
        except NoCredentialsError: 
            logger.error("S3 credentials not available for upload.")
            return None
        except ClientError as e:
            logger.error(f"S3 ClientError during upload of '{local_file_path}' to '{object_key}': {e}")
            return None
        except Exception as e: 
            logger.error(f"Unexpected error during upload of '{local_file_path}' to '{object_key}': {e}")
            return None

    def _download_from_spaces(self, object_key: str, local_file_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, object_key, local_file_path)
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{object_key} to {local_file_path}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.error(f"Object not found in Spaces: s3://{self.bucket_name}/{object_key}")
            else:
                logger.error(f"S3 ClientError during download of '{object_key}': {e}")
            return False
        except Exception as e: 
            logger.error(f"Unexpected error during download of '{object_key}': {e}")
            return False
            
    def _download_file_from_url(self, file_url: str, local_file_path: str) -> bool:
        """Downloads a file from a public HTTPs URL."""
        headers = { 
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': '*/*', 
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        try:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            logger.info(f"Attempting to download file from URL: {file_url} to {local_file_path}")
            with requests.get(file_url, stream=True, headers=headers, timeout=120) as r: 
                logger.info(f"Response status code for {file_url}: {r.status_code}")
                r.raise_for_status() 
                with open(local_file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            logger.info(f"Successfully downloaded file from {file_url} to {local_file_path}")
            return True
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code if e.response else 'Unknown Status'} during download from URL '{file_url}': {e}")
            if e.response is not None:
                logger.error(f"Response headers: {e.response.headers}")
                logger.error(f"Response content (first 500 chars): {e.response.text[:500]}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout during download from URL '{file_url}'")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during download from URL '{file_url}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download from URL '{file_url}': {e}")
            return False

    def _delete_prefix_in_spaces(self, prefix: str):
        if not prefix.endswith('/'): 
            prefix += '/'
        logger.info(f"Attempting to list objects under prefix for deletion: s3://{self.bucket_name}/{prefix}")
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            objects_to_delete = []
            try:
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            objects_to_delete.append({'Key': obj['Key']})
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                    logger.info(f"No objects found under prefix (or bucket issue): {prefix}. Error: {e}")
                else:
                    raise 
            if not objects_to_delete:
                logger.info(f"No objects to delete under prefix: {prefix}")
                return
            for i in range(0, len(objects_to_delete), 1000): 
                chunk = objects_to_delete[i:i+1000]
                delete_payload = {'Objects': chunk}
                self.s3_client.delete_objects(Bucket=self.bucket_name, Delete=delete_payload)
                logger.info(f"Deleted {len(chunk)} objects under prefix {prefix}")
            logger.info(f"Finished deleting objects under prefix: {prefix}")
        except ClientError as e:
            logger.error(f"S3 ClientError during prefix deletion operation for '{prefix}': {e}")
        except Exception as e: 
            logger.error(f"Unexpected error during prefix deletion for '{prefix}': {e}")

    def clean_spaces_output_directories(self):
        logger.info(f"Clearing previous results from DigitalOcean Spaces under base prefix: {self.base_spaces_output_prefix}")
        self._delete_prefix_in_spaces(self.base_spaces_output_prefix)

    def save_uncompressed_copy_to_spaces(self, source_local_path: str, target_uncompressed_object_key: str) -> Optional[str]:
        if not os.path.exists(source_local_path):
            logger.error(f"Source file for uncompression not found: {source_local_path}")
            return None
        temp_uncompressed_local_filename = "temp_uncompressed_" + os.path.basename(target_uncompressed_object_key)
        temp_uncompressed_local_path = os.path.join(self.temp_local_dir, temp_uncompressed_local_filename)
        try:
            img = nib.load(source_local_path)
            uncompressed_img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
            nib.save(uncompressed_img, temp_uncompressed_local_path)
            logger.info(f"Saved temporary local uncompressed copy at: {temp_uncompressed_local_path}")
            uploaded_s3_uri = self._upload_to_spaces(temp_uncompressed_local_path, target_uncompressed_object_key)
            return uploaded_s3_uri
        except Exception as e:
            logger.error(f"Error saving uncompressed copy of '{source_local_path}' to Spaces as '{target_uncompressed_object_key}': {e}")
            return None
        finally:
            if os.path.exists(temp_uncompressed_local_path):
                try:
                    os.remove(temp_uncompressed_local_path)
                except Exception as e_rem: 
                    logger.warning(f"Could not remove temp uncompressed file {temp_uncompressed_local_path}: {e_rem}")


    def run_pipeline(self) -> Dict:
        # --- Atlas Handling: Download from CDN URLs ---
        atlas_base_cdn_url = "https://nifti-bucket.lon1.cdn.digitaloceanspaces.com"
        
        atlas_urls = {
            'cortex': f"{atlas_base_cdn_url}/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
            'subcortex': f"{atlas_base_cdn_url}/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
            'mni_atlas': f"{atlas_base_cdn_url}/MNI-maxprob-thr25-1mm.nii.gz"
        }
        mni_template_url = f"{atlas_base_cdn_url}/MNI152_T1_1mm_brain.nii.gz"

        logger.info("Downloading common atlas and template files via HTTPs...")
        
        mni_template_filename = os.path.basename(mni_template_url)
        local_mni_template_path = os.path.join(self.local_common_atlases_dir, mni_template_filename)
        if not self._download_file_from_url(mni_template_url, local_mni_template_path): 
            raise RuntimeError(f"CRITICAL: Failed to download MNI template from URL: {mni_template_url}")

        downloaded_atlas_local_paths = {}
        for atlas_name_key, url in atlas_urls.items():
            atlas_filename = os.path.basename(url)
            local_path = os.path.join(self.local_common_atlases_dir, atlas_filename)
            if not self._download_file_from_url(url, local_path): 
                raise RuntimeError(f"CRITICAL: Failed to download atlas '{atlas_filename}' from URL: {url}")
            downloaded_atlas_local_paths[atlas_name_key] = local_path
        
        atlas_paths_for_pipeline = {
            'cortex': downloaded_atlas_local_paths['cortex'],
            'subcortex': downloaded_atlas_local_paths['subcortex'],
            'mni': downloaded_atlas_local_paths['mni_atlas']
        }
        # --- End of Atlas Download Handling ---

        self.clean_spaces_output_directories()
        
        local_input_filename = os.path.basename(self.input_object_key)
        local_input_file_path = os.path.join(self.local_uploads_dir, local_input_filename) 
        
        if not self._download_from_spaces(self.input_object_key, local_input_file_path):
            msg = f"CRITICAL: Failed to download input file: {self.input_object_key} from bucket {self.bucket_name}"
            logger.error(msg)
            raise RuntimeError(msg)

        local_brain_path = os.path.join(self.local_processed_dir, "brain.nii.gz")
        local_norm_path = os.path.join(self.local_processed_dir, "normalized.nii.gz")
        local_bias_corrected_path = os.path.join(self.local_processed_dir, "bias_corrected.nii.gz")
        local_registered_path = os.path.join(self.local_processed_dir, "registered.nii.gz") 

        output_s3_uris = {
            "input_original_s3_uri": f"s3://{self.bucket_name}/{self.input_object_key}",
            "input_uncompressed_s3_uri": None,
            "brain_extracted_compressed_s3_uri": None,
            "brain_extracted_uncompressed_s3_uri": None,
            "normalized_compressed_s3_uri": None,
            "normalized_uncompressed_s3_uri": None,
            "bias_corrected_compressed_s3_uri": None,
            "bias_corrected_uncompressed_s3_uri": None,
            "registered_compressed_s3_uri": None,
            "registered_uncompressed_s3_uri": None,
            "segmented_files_uncompressed_s3_uris": {}, 
            "volumetrics_cortex_csv_s3_uri": None,
            "volumetrics_subcortex_csv_s3_uri": None,
            "volumetrics_mni_csv_s3_uri": None,
            "volumetric_data_json": {} 
        }

        try:
            logger.info(f"Starting MRI processing pipeline with Spaces for input: {self.input_object_key}")

            input_uncompressed_filename = local_input_filename.replace(".nii.gz", ".nii") if local_input_filename.endswith(".nii.gz") else local_input_filename + ".nii"
            input_uncompressed_key = os.path.join(self.uncompressed_spaces_prefix, input_uncompressed_filename)
            output_s3_uris["input_uncompressed_s3_uri"] = self.save_uncompressed_copy_to_spaces(local_input_file_path, input_uncompressed_key)

            logger.info(f"➡ Extracting brain from {local_input_file_path} to save as 'brain.nii.gz' in {self.local_processed_dir}")
            returned_brain_path = self.pipeline.extract_brain_part(local_input_file_path, "brain.nii.gz") 
            if not (returned_brain_path and os.path.exists(returned_brain_path) and returned_brain_path == local_brain_path):
                logger.error(f"Brain extraction did not produce expected file. Expected: {local_brain_path}, Got: {returned_brain_path}")
                raise FileNotFoundError(f"Brain extraction output not found or path mismatch: expected {local_brain_path}")
            output_s3_uris["brain_extracted_compressed_s3_uri"] = self._upload_to_spaces(local_brain_path, os.path.join(self.processed_spaces_prefix, "brain.nii.gz"))
            output_s3_uris["brain_extracted_uncompressed_s3_uri"] = self.save_uncompressed_copy_to_spaces(local_brain_path, os.path.join(self.uncompressed_spaces_prefix, "brain.nii"))
            
            logger.info(f"➡ Normalizing intensity for {local_brain_path}, output to {local_norm_path}")
            brain_image = nib.load(local_brain_path)
            brain_data = brain_image.get_fdata()
            norm_data = self.pipeline.intensity_normalisation(brain_data)
            nib.save(nib.Nifti1Image(norm_data, brain_image.affine, brain_image.header), local_norm_path)
            if not os.path.exists(local_norm_path): raise FileNotFoundError(f"Normalization output not found: {local_norm_path}")
            output_s3_uris["normalized_compressed_s3_uri"] = self._upload_to_spaces(local_norm_path, os.path.join(self.processed_spaces_prefix, "normalized.nii.gz"))
            output_s3_uris["normalized_uncompressed_s3_uri"] = self.save_uncompressed_copy_to_spaces(local_norm_path, os.path.join(self.uncompressed_spaces_prefix, "normalized.nii"))

            logger.info(f"➡ Applying bias correction to {local_norm_path}, output to {local_bias_corrected_path}")
            returned_bias_path = self.pipeline.fsl_bias_correction(local_norm_path, local_bias_corrected_path)
            if not (returned_bias_path and os.path.exists(returned_bias_path) and returned_bias_path == local_bias_corrected_path):
                logger.error(f"Bias correction did not produce expected file. Expected: {local_bias_corrected_path}, Got: {returned_bias_path}")
                raise FileNotFoundError(f"Bias correction output not found or path mismatch: expected {local_bias_corrected_path}")
            output_s3_uris["bias_corrected_compressed_s3_uri"] = self._upload_to_spaces(local_bias_corrected_path, os.path.join(self.processed_spaces_prefix, "bias_corrected.nii.gz"))
            output_s3_uris["bias_corrected_uncompressed_s3_uri"] = self.save_uncompressed_copy_to_spaces(local_bias_corrected_path, os.path.join(self.uncompressed_spaces_prefix, "bias_corrected.nii"))

            logger.info(f"➡ Registering {local_bias_corrected_path} to MNI template: {local_mni_template_path}. Output dir for registration: {self.local_processed_dir}")
            returned_registered_path_from_method = self.pipeline.image_registration_mni(local_bias_corrected_path, local_mni_template_path, self.local_processed_dir)
            
            if not (returned_registered_path_from_method and os.path.exists(returned_registered_path_from_method) and returned_registered_path_from_method == local_registered_path):
                logger.error(f"Registration output path mismatch or file not found. Expected: {local_registered_path}, Actual from method: {returned_registered_path_from_method}")
                if os.path.exists(self.local_processed_dir):
                    logger.info(f"Contents of {self.local_processed_dir} after registration attempt: {os.listdir(self.local_processed_dir)}")
                else:
                    logger.info(f"Directory {self.local_processed_dir} does not exist after registration attempt.")
                raise FileNotFoundError(f"Registered file not found at expected location: {local_registered_path} (or method returned different/invalid path).")
            
            output_s3_uris["registered_compressed_s3_uri"] = self._upload_to_spaces(local_registered_path, os.path.join(self.processed_spaces_prefix, "registered.nii.gz"))
            output_s3_uris["registered_uncompressed_s3_uri"] = self.save_uncompressed_copy_to_spaces(local_registered_path, os.path.join(self.uncompressed_spaces_prefix, "registered.nii"))

            logger.info(f"➡ Segmenting brain images using downloaded atlases. Input dir for segmenter: {self.local_processed_dir}, Output dir for segmenter: {self.local_segmented_dir}")
            segmenter = ImageSegmenter_class(
                images_dir=self.local_processed_dir, 
                atlas_paths=atlas_paths_for_pipeline,
                output_dir=self.local_segmented_dir
                # No longer passing bet_executable_path
            )
            files_to_segment = ["registered.nii.gz"] 
            for file_to_seg in files_to_segment:
                segmenter.process_single_image((file_to_seg, 'all')) 
            
            for root, _, files in os.walk(self.local_segmented_dir):
                for file in files:
                    if file.endswith('.nii.gz') or file.endswith('.nii'): 
                        local_segmented_file_path = os.path.join(root, file)
                        relative_path_in_segmentation = os.path.relpath(local_segmented_file_path, self.local_segmented_dir)
                        spaces_segmented_key = os.path.join(self.segmented_spaces_prefix, relative_path_in_segmentation)
                        self._upload_to_spaces(local_segmented_file_path, spaces_segmented_key) 
                        
                        if file.endswith('.nii.gz'):
                            uncompressed_filename_base = relative_path_in_segmentation.replace(".nii.gz", ".nii")
                            uncompressed_folder_name = os.path.basename(os.path.dirname(spaces_segmented_key)) 
                            spaces_segmented_uncompressed_key = os.path.join(self.uncompressed_spaces_prefix, uncompressed_folder_name, os.path.basename(uncompressed_filename_base))
                            dict_key_for_uncompressed = f"{uncompressed_folder_name}_{os.path.basename(uncompressed_filename_base).replace('.nii','')}"
                            output_s3_uris["segmented_files_uncompressed_s3_uris"][dict_key_for_uncompressed] = self.save_uncompressed_copy_to_spaces(local_segmented_file_path, spaces_segmented_uncompressed_key)

            logger.info(f"➡ Extracting volumetric features. Segmented images dir: {self.local_segmented_dir}, Output CSVs to: {self.local_features_dir}")
            feature_extractor = FeatureExtraction_class(
                images_dir=self.local_segmented_dir, 
                atlas_paths=atlas_paths_for_pipeline, 
                output_dir=self.local_features_dir
                # No longer passing bet_executable_path
            )
            
            cortex_csv_basename = "volumetrics_cortex.csv"
            subcortex_csv_basename = "volumetrics_subcortex.csv"
            mni_csv_basename = "volumetrics_mni.csv"

            cortex_vol_data = feature_extractor.extract_features(os.path.join(self.local_segmented_dir, "segmented_cortex"), cortex_csv_basename)
            subcortex_vol_data = feature_extractor.extract_features(os.path.join(self.local_segmented_dir, "segmented_subcortex"), subcortex_csv_basename)
            mni_vol_data = feature_extractor.extract_features(os.path.join(self.local_segmented_dir, "segmented_mni"), mni_csv_basename)

            output_s3_uris["volumetric_data_json"] = {
                "cortex": cortex_vol_data,
                "subcortex": subcortex_vol_data,
                "mni": mni_vol_data
            }

            output_s3_uris["volumetrics_cortex_csv_s3_uri"] = self._upload_to_spaces(os.path.join(self.local_features_dir, cortex_csv_basename), os.path.join(self.features_spaces_prefix, cortex_csv_basename))
            output_s3_uris["volumetrics_subcortex_csv_s3_uri"] = self._upload_to_spaces(os.path.join(self.local_features_dir, subcortex_csv_basename), os.path.join(self.features_spaces_prefix, subcortex_csv_basename))
            output_s3_uris["volumetrics_mni_csv_s3_uri"] = self._upload_to_spaces(os.path.join(self.local_features_dir, mni_csv_basename), os.path.join(self.features_spaces_prefix, mni_csv_basename))

            logger.info("MRI processing pipeline completed successfully.")
            return output_s3_uris

        except FileNotFoundError as fnf_error:
            logger.error(f"FileNotFoundError during pipeline: {fnf_error}", exc_info=True)
            self._cleanup_temp_dir() 
            raise RuntimeError(f"Pipeline failed due to missing file: {fnf_error}") from fnf_error
        except Exception as e:
            logger.error(f"Error during MRI processing pipeline: {e}", exc_info=True)
            self._cleanup_temp_dir() 
            raise RuntimeError(f"Pipeline failed with error: {e}") from e 

    def _cleanup_temp_dir(self):
        if self.temp_local_dir and os.path.exists(self.temp_local_dir): 
            try:
                shutil.rmtree(self.temp_local_dir)
                logger.info(f"Successfully removed temporary directory: {self.temp_local_dir}")
            except Exception as e:
                logger.error(f"Error removing temporary directory {self.temp_local_dir}: {e}")
        else:
            logger.info("Temporary directory was not found or already cleaned up (or init failed before its creation).")

