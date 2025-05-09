import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ants
import subprocess
import os
import logging 
import tempfile 
import shutil 

from nipype.interfaces import fsl 
from matplotlib.colors import LinearSegmentedColormap
from nibabel.processing import resample_from_to
from concurrent.futures import ProcessPoolExecutor
from skimage import measure, morphology, filters
from scipy import ndimage 
from scipy.spatial import ConvexHull 
from scipy.linalg import svd
from fsl.wrappers.bet import bet as fsl_bet_wrapper # Import the wrapper

# Configure basic logging
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRIScansPipeline():
    def __init__(self, input_dir, output_dir): # bet_executable_path is no longer needed
        self.input = input_dir
        self.output = output_dir
        os.makedirs(self.input, exist_ok=True)
        os.makedirs(self.output, exist_ok=True)
        logger.info(f"MRIScansPipeline initialized. Input: {self.input}, Output: {self.output}. Using fsl.wrappers.bet.")
        
    def display_mri_image(self, input_image_path):
        logger.info(f"Loading MRI image for display: {input_image_path}")
        if not os.path.exists(input_image_path):
            logger.error(f"MRI image not found at {input_image_path}")
            return None, None
            
        mri_scan = nib.load(input_image_path)
        mri_data = mri_scan.get_fdata()
        
        logger.info(f"Image Dimensions: {mri_scan.shape}")
        logger.info(f"Voxel Dimensions: {mri_scan.header.get_zooms()}")
        # ... (rest of display logic, avoiding plt.show() for server)
        return mri_data, mri_scan
        
    def extract_brain_part(self, input_image_path, output_filename):
        """Performs skull stripping using fsl.wrappers.bet.bet.
        
        Args:   
           input_image_path (str): Path to the input MRI scan.
           output_filename (str): Filename for the extracted brain part (will be saved in self.output).
           
        Raises:
            RuntimeError: If BET process fails.
        """
        full_output_path = os.path.join(self.output, output_filename)
        logger.info(f"Attempting brain extraction using fsl.wrappers.bet.bet. Input: {input_image_path}, Output: {full_output_path}")

        try:
            # MODIFIED: Call fsl_bet_wrapper with positional arguments for input and output
            fsl_bet_wrapper(input_image_path, 
                            full_output_path, 
                            robust=True) # -R option

            if not os.path.exists(full_output_path):
                error_message = f"BET output file not found at {full_output_path} after running fsl.wrappers.bet.bet."
                logger.error(error_message)
                raise FileNotFoundError(error_message)
            
            logger.info(f"Brain extraction successful using fsl.wrappers.bet.bet. Output: {full_output_path}")
            return full_output_path
            
        except Exception as e:
            error_message = f"BET (using fsl.wrapper) failed for input {input_image_path}!\nError: {str(e)}"
            logger.error(error_message, exc_info=True) 
            raise RuntimeError(error_message) from e
        
    def intensity_normalisation(self, image_data):
        logger.info("Normalizing intensity.")
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        if max_val == min_val:
            logger.warning("Image data has uniform intensity. Normalization will result in a zero image (or original if min_val is subtracted).")
            return image_data - min_val 
        normalised_image_data = (image_data - min_val) / (max_val - min_val)
        return normalised_image_data
    
    def fsl_bias_correction(self, input_image_path, output_path):
        logger.info(f"Starting FSL FAST bias correction for {input_image_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fast_out_dir = None 
        try:
            fast_interface = fsl.FAST()
            fast_interface.inputs.in_files = input_image_path
            fast_interface.inputs.bias_iters = 5 
            fast_interface.inputs.bias_lowpass = 20 
            fast_interface.inputs.output_biascorrected = True 
            fast_interface.inputs.img_type = 1 
            
            fast_out_dir = tempfile.mkdtemp(dir=self.output, prefix="fast_temp_")
            base_name = os.path.basename(input_image_path).replace(".nii.gz", "")
            fast_interface.inputs.out_basename = os.path.join(fast_out_dir, f"{base_name}_fast_corrected")

            logger.info(f"Running FSL FAST. Output base for FAST internal files: {fast_interface.inputs.out_basename}")
            result = fast_interface.run() 
            
            corrected_file_found = None
            if hasattr(result.outputs, 'restored_image') and result.outputs.restored_image and os.path.exists(result.outputs.restored_image):
                corrected_file_found = result.outputs.restored_image
            elif hasattr(result.outputs, 'bias_corrected_image') and result.outputs.bias_corrected_image and os.path.exists(result.outputs.bias_corrected_image):
                 corrected_file_found = result.outputs.bias_corrected_image
            else: 
                expected_corrected_file = os.path.join(fast_out_dir, f"{base_name}_fast_corrected_restore.nii.gz")
                if os.path.exists(expected_corrected_file):
                    corrected_file_found = expected_corrected_file
            
            if not corrected_file_found or not os.path.exists(corrected_file_found):
                logger.error(f"FSL FAST output files in {fast_out_dir}: {os.listdir(fast_out_dir) if os.path.exists(fast_out_dir) else 'FAST output directory does not exist or is empty'}")
                raise FileNotFoundError(f"Expected bias-corrected file not found after FAST. Check Nipype FAST output naming and paths.")

            logger.info(f"Bias corrected file found at: {corrected_file_found}. Moving to final output path: {output_path}")
            shutil.move(corrected_file_found, output_path)

            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Bias field corrected file not found after move: {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error during FSL FAST bias correction for {input_image_path}: {e}", exc_info=True)
            raise RuntimeError(f"FSL FAST bias correction failed: {e}")
        finally:
            if fast_out_dir and os.path.exists(fast_out_dir):
                shutil.rmtree(fast_out_dir)
                logger.info(f"Cleaned up temporary FAST output directory: {fast_out_dir}")
        
    def image_registration_mni(self, moving_image_path, template_mni_path, output_prefix_dir):
        logger.info(f"Starting ANTs registration: moving='{moving_image_path}', fixed='{template_mni_path}'")
        try:
            fixed = ants.image_read(template_mni_path)
            moving = ants.image_read(moving_image_path)
            
            registration = ants.registration(
                fixed=fixed, moving=moving, type_of_transform='SyN', 
                grad_step=0.2, flow_sigma=3, total_sigma=0, 
                aff_metric='mattes', syn_metric='mattes',
                verbose=False 
            )
            
            os.makedirs(output_prefix_dir, exist_ok=True) 
            registered_image_path = os.path.join(output_prefix_dir, 'registered.nii.gz')
            
            warped_img = registration['warpedmovout']
            ants.image_write(warped_img, registered_image_path)
            logger.info(f"ANTs registration successful. Output: {registered_image_path}")
            return registered_image_path
        except Exception as e:
            logger.error(f"Error during ANTs registration: {str(e)}", exc_info=True)
            return None
        
class BatchProcessingImages(MRIScansPipeline): 
    def __init__(self, input_dir, output_dir, registered_dir): 
        super().__init__(input_dir, output_dir) 
        self.registered = registered_dir 
        os.makedirs(self.registered, exist_ok=True)
        
    def process_mri_image(self, args):
        patient_id, image_id, mni_path = args
        logger.info(f"Processing image: {patient_id}/{image_id}")
        try:
            patient_input_path = os.path.join(self.input, patient_id)
            image_path = os.path.join(patient_input_path, image_id)
            
            if not os.path.isfile(image_path):
                logger.warning(f"File not found: {image_path}")
                return f"File not found: {image_path}"
            
            base_filename = f"{patient_id}-{image_id.replace('.nii.gz','')}"
            final_registered_image_path = os.path.join(self.registered, f"{base_filename}-registered.nii.gz")
            
            if os.path.exists(final_registered_image_path):
                logger.info(f"Registered image already exists, skipping: {final_registered_image_path}")
                return f"Already exists: {final_registered_image_path}"
            
            brain_extracted_filename = f"{base_filename}-brain_extracted.nii.gz"
            brain_extracted_path = self.extract_brain_part(image_path, brain_extracted_filename)

            brain_extracted_img = nib.load(brain_extracted_path)
            brain_extracted_data = brain_extracted_img.get_fdata()
            normalised_image_data = self.intensity_normalisation(brain_extracted_data)
            
            normalised_image = nib.Nifti1Image(normalised_image_data, brain_extracted_img.affine)
            normalised_image_filename = f"{base_filename}-normalized.nii.gz"
            normalised_image_path = os.path.join(self.output, normalised_image_filename)
            nib.save(normalised_image, normalised_image_path)
            logger.info(f"Normalized image saved to: {normalised_image_path}")
            
            bias_corrected_filename = f"{base_filename}-bias_corrected.nii.gz"
            bias_corrected_path = os.path.join(self.output, bias_corrected_filename)
            bias_corrected_path = self.fsl_bias_correction(normalised_image_path, bias_corrected_path)
            logger.info(f"Bias corrected image saved to: {bias_corrected_path}")
            
            temp_registration_output_dir = tempfile.mkdtemp(dir=self.output, prefix=f"reg_temp_{base_filename}_")
            registered_temp_path = self.image_registration_mni(bias_corrected_path, mni_path, temp_registration_output_dir)
            
            if registered_temp_path and os.path.exists(registered_temp_path):
                shutil.move(registered_temp_path, final_registered_image_path)
                logger.info(f"Registered image moved to: {final_registered_image_path}")
                shutil.rmtree(temp_registration_output_dir)
                return f"Processing complete: {final_registered_image_path}"
            else:
                if os.path.exists(temp_registration_output_dir): shutil.rmtree(temp_registration_output_dir)
                logger.error(f"Registration failed for {image_path}")
                return f"Registration failed for {patient_id}-{image_id}"
        except Exception as e:
            logger.error(f"Error processing {patient_id}-{image_id}: {str(e)}", exc_info=True)
            return f"Error processing {patient_id}-{image_id}: {str(e)}"
        
    def batch_process_mri_images(self, mni_path, num_workers=4):
        tasks = []
        for patient_id in os.listdir(self.input):
            patient_path = os.path.join(self.input, patient_id)
            if not os.path.isdir(patient_path) or patient_id == '.DS_Store':
                continue
            patient_images = [file for file in os.listdir(patient_path) if file.endswith('.nii.gz')]
            if not patient_images:
                logger.info(f"No MRI images found for {patient_id}")
                continue
            for image_id in patient_images:
                tasks.append((patient_id, image_id, mni_path))
        if not tasks:
            logger.info("No tasks to process for batch_process_mri_images.")
            return
        logger.info(f"Starting batch processing of {len(tasks)} images with {num_workers} workers.")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.process_mri_image, tasks))
        for result in results:
            logger.info(result)
        logger.info("Batch processing completed successfully!")  
        
class ImageSegmenter(BatchProcessingImages):
    def __init__(self, images_dir, atlas_paths, output_dir): 
        super().__init__(input_dir=images_dir, output_dir=output_dir, registered_dir=images_dir) 
        
        labels_subcortex = {
            "left_thalamus": 4, "right_thalamus": 15, "left_hippocampus": 9, "right_hippocampus": 19,
            "left_amygdala": 10, "right_amygdala": 20, "left_cortex": 2, "right_cortex": 13,
            "left_lateral_ventricle": 3, "right_lateral_ventricle": 14, "left_caudate": 5, "right_caudate": 16,
            "left_putamen": 6, "right_putamen": 17
        }
        labels_cortex = {
            "frontal_pole": 1, "temporal_pole": 8, "superior_frontal_gyrus": 3, "middle_frontal_gyrus": 4,
            "superior_parietal_lobule": 18, "frontal_medial_cortex": 25, "parahippocampal_gyrus_anterior": 34,
            "parahippocampal_gyrus_posterior": 35, "occipital_pole": 48, "angular_gyrus": 21,
            "frontal_orbital_cortex": 33, "lateral_occipital_cortex_inferior": 23, "lateral_occipital_cortex_superior": 22,
        }    
        mni_labels= {
            "frontal_lobe": 3, "occipital_lobe": 5, "parietal_lobe": 6, "temporal_lobe": 8
        }
        self.images_dir = images_dir 
        self.atlas_paths = atlas_paths 
        self.output_dir = output_dir 
        
        self.label_mappings = {
            'cortex': labels_cortex,
            'subcortex': labels_subcortex,
            'mni': mni_labels
        }
        
        for seg_type in self.label_mappings.keys():
            os.makedirs(os.path.join(self.output_dir, f"segmented_{seg_type}"), exist_ok=True)
        logger.info(f"ImageSegmenter initialized. Outputting segmented files to subdirectories of: {self.output_dir}")

    def process_single_image(self, args):
        file, segmentation_type = args 
        logger.info(f"Segmenting image: {file} for type: {segmentation_type}")
        try:
            if not file.endswith('.nii.gz'):
                logger.warning(f"Skipped non-nii.gz file: {file}")
                return f"Skipped non-nii.gz file: {file}"
            registered_mri_path = os.path.join(self.images_dir, file) 
            if not os.path.exists(registered_mri_path):
                logger.error(f"Registered MRI file not found for segmentation: {registered_mri_path}")
                return f"Registered MRI file not found: {registered_mri_path}"
            registered_mri_image = nib.load(registered_mri_path)
            registered_mri_data = registered_mri_image.get_fdata()
            segmentations_to_perform = list(self.label_mappings.keys()) if segmentation_type == 'all' else [segmentation_type]
            for seg_type in segmentations_to_perform:
                if seg_type not in self.atlas_paths:
                    logger.warning(f"Atlas path for segmentation type '{seg_type}' not found. Skipping.")
                    continue
                if seg_type not in self.label_mappings:
                    logger.warning(f"Label mapping for segmentation type '{seg_type}' not found. Skipping.")
                    continue
                atlas_path = self.atlas_paths[seg_type]
                if not os.path.exists(atlas_path):
                    logger.error(f"Atlas file not found: {atlas_path} for seg_type: {seg_type}")
                    continue
                atlas_img = nib.load(atlas_path)
                atlas_data = atlas_img.get_fdata()
                current_labels = self.label_mappings[seg_type]
                output_subdir_for_seg_type = os.path.join(self.output_dir, f"segmented_{seg_type}")
                base_input_filename = file.replace("_registered.nii.gz", "").replace(".nii.gz", "")

                for label_name, label_value in current_labels.items():
                    label_out_filename = f"{base_input_filename}_{label_name}.nii.gz" 
                    label_out_path = os.path.join(output_subdir_for_seg_type, label_out_filename)
                    
                    if os.path.exists(label_out_path):
                        logger.info(f"Segmented output {label_out_path} already exists. Skipping.")
                        continue
                    label_mask = np.isin(atlas_data, label_value).astype(np.float32)
                    label_mask_img = nib.Nifti1Image(label_mask, atlas_img.affine)
                    resampled_mask_img = resample_from_to(label_mask_img, registered_mri_image, order=0)
                    label_data = registered_mri_data * resampled_mask_img.get_fdata()
                    label_img = nib.Nifti1Image(label_data, registered_mri_image.affine)
                    nib.save(label_img, label_out_path)
                    logger.info(f"Saved segmented label {label_name} for {file} to {label_out_path}")
            return f"Successfully processed {file} for segmentation type(s): {', '.join(segmentations_to_perform)}"
        except Exception as e:
            logger.error(f"Error processing {file} for segmentation: {str(e)}", exc_info=True)
            return f"Error processing {file} for segmentation: {str(e)}"

    def batch_segment_images(self, segmentation_type='all', num_workers=4):
        tasks = []
        files = [f for f in os.listdir(self.images_dir) if f.endswith('.nii.gz') and 'registered' in f]
        for file in files:
            tasks.append((file, segmentation_type))
        if not tasks:
            logger.info(f"No registered NIFTI files found in {self.images_dir} to segment.")
            return
        logger.info(f"Starting batch segmentation of {len(tasks)} files with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.process_single_image, tasks))
        for result in results:
            logger.info(result)
        logger.info("Batch segmentation completed!")
        
    def plot_segmented_img(self, bg_img_path, segmented_img_path):
        logger.info(f"Plotting segmented image: {segmented_img_path} on background: {bg_img_path}")
        # ... (plotting logic, avoid plt.show())

class FeatureExtraction(ImageSegmenter):
    def __init__(self, images_dir, atlas_paths, output_dir): 
        super().__init__(images_dir, atlas_paths, output_dir) 
        logger.info(f"FeatureExtraction initialized. Outputting CSVs to: {self.output_dir}")
        
    def extract_volume(self, segmented_img_path):
        img = nib.load(segmented_img_path)
        data = img.get_fdata()
        voxel_size = np.prod(img.header.get_zooms())
        volume = np.sum(data > 1e-5) * voxel_size 
        return volume
    
    def extract_surface_area(self, segmented_image_path, method="greyscale", percentile=50):
        img = nib.load(segmented_image_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()
        if data.max() < 1e-5: return 0.0 
        if method == 'binary':
            threshold = filters.threshold_otsu(data[data > 1e-5]) if len(data[data > 1e-5]) > 0 else 0.5
            data_processed = data > threshold
            level = 0.5 
        else: 
            data_processed = data
            non_zero_data = data[data > 1e-5]
            level = np.percentile(non_zero_data, percentile) if len(non_zero_data) > 0 else data.mean()
        try:
            verts, faces, _, _ = measure.marching_cubes(data_processed, level=level, spacing=spacing)
            return measure.mesh_surface_area(verts, faces)
        except Exception as e:
            logger.warning(f"Could not calculate surface area for {segmented_image_path} (level={level}): {e}. Returning 0.")
            return 0.0
    
    def extract_compactness(self, volume, surface_area):
        if volume <= 1e-9 or surface_area <= 1e-9: return 0.0
        return (36 * np.pi * volume**2) / (surface_area**3) if surface_area > 1e-9 else 0.0

    def extract_sphericity(self, volume, surface_area):
        if surface_area <= 1e-9 or volume <= 1e-9: return 0.0
        return (np.pi**(1/3)) * (6 * volume)**(2/3) / surface_area
    
    def extract_eccentricity(self, segmented_image_path):
        img = nib.load(segmented_image_path)
        data = img.get_fdata()
        if data.max() < 1e-5: return 0.0
        spacing = img.header.get_zooms()
        coords = np.array(np.where(data > 1e-5)).T * spacing
        if len(coords) < 4:
            logger.warning(f"Not enough points ({len(coords)}) for ellipsoid fitting in {segmented_image_path}")
            return 0.0
        if coords.shape[0] <= coords.shape[1]:
             logger.warning(f"Too few points ({coords.shape[0]}) for robust covariance in {segmented_image_path}")
             return 0.0
        cov_matrix = np.cov(coords, rowvar=False)
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            logger.warning(f"NaN or Inf in covariance matrix for {segmented_image_path}.")
            return 0.0
        try:
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1] 
            if eigenvalues[0] <= 1e-9: return 0.0
            if eigenvalues[2] < 0 or eigenvalues[0] < 0: 
                 logger.warning(f"Negative eigenvalues for {segmented_image_path}: {eigenvalues}.")
                 return 0.0
            # Ensure eigenvalues[0] is not zero before division
            if eigenvalues[0] == 0:
                logger.warning(f"Major eigenvalue is zero for {segmented_image_path}. Cannot compute eccentricity.")
                return 0.0
            eccentricity_sq_term = eigenvalues[2] / eigenvalues[0]
            if eccentricity_sq_term > 1: # Should not happen if sorted correctly and positive
                logger.warning(f"Minor/Major eigenvalue ratio > 1 ({eccentricity_sq_term}) for {segmented_image_path}. Clamping.")
                eccentricity_sq_term = 1.0
            
            eccentricity = np.sqrt(1 - eccentricity_sq_term)
            
            if not (0 <= eccentricity <= 1): 
                logger.warning(f"Invalid eccentricity {eccentricity} for {segmented_image_path}. Eigenvalues: {eigenvalues}")
                return 0.0 
            return eccentricity
        except np.linalg.LinAlgError as e: 
            logger.error(f"LinAlgError calculating eccentricity for {segmented_image_path}: {str(e)}.")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating eccentricity for {segmented_image_path}: {str(e)}", exc_info=True)
            return 0.0
    
    def extract_features(self, segmented_dir, output_csv_filename=None):
        files = os.listdir(segmented_dir)
        logger.info(f"Extracting features from directory: {segmented_dir}")
        current_scan_features = {}
        for file in files:
            if not file.endswith('.nii.gz'):
                continue
            part_name = file.replace('.nii.gz', '')
            image_path = os.path.join(segmented_dir, file)
            if os.path.exists(image_path):
                logger.info(f"Processing features for: {image_path}")
                try:
                    volume = self.extract_volume(image_path)
                    surface_area = self.extract_surface_area(image_path)
                    compactness = self.extract_compactness(volume, surface_area)
                    sphericity = self.extract_sphericity(volume, surface_area)
                    eccentricity = self.extract_eccentricity(image_path)
                    current_scan_features[f"{part_name}_Volume"] = float(volume)
                    current_scan_features[f"{part_name}_Surface_Area"] = float(surface_area)
                    current_scan_features[f"{part_name}_Compactness"] = float(compactness)
                    current_scan_features[f"{part_name}_Sphericity"] = float(sphericity)
                    current_scan_features[f"{part_name}_Eccentricity"] = float(eccentricity)
                except Exception as e:
                    logger.error(f"Error processing features for file {image_path}: {e}", exc_info=True)
                    current_scan_features[f"{part_name}_Volume"] = np.nan 
                    current_scan_features[f"{part_name}_Surface_Area"] = np.nan
                    current_scan_features[f"{part_name}_Compactness"] = np.nan
                    current_scan_features[f"{part_name}_Sphericity"] = np.nan
                    current_scan_features[f"{part_name}_Eccentricity"] = np.nan
                    continue  
        if output_csv_filename:
            features_df = pd.DataFrame([current_scan_features]) 
            csv_full_path = os.path.join(self.output_dir, output_csv_filename) 
            features_df.to_csv(csv_full_path, index=False)
            logger.info(f"Features extracted and saved to: {csv_full_path}")
        return current_scan_features

# The if __name__ == "__main__": block from user's MRIScansPipeline.py is omitted here
# as this file is intended to be imported as a module.
