import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ants
import subprocess
import os
from nipype.interfaces import fsl, elastix
from matplotlib.colors import LinearSegmentedColormap
from nibabel.processing import resample_from_to
from concurrent.futures import ProcessPoolExecutor
from skimage import measure, morphology, filters
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.linalg import svd

class MRIScansPipeline():
    def __init__(self, input_dir, output_dir):
        self.input = input_dir
        self.output = output_dir
        
    def display_mri_image(self, input_image_path):
        """Load and display slices from an MRI scan (NIfTI format). using Nibabel library
        
        Args:
            input_image_path (str): path of the imput image
        """
        
        mri_scan = nib.load(input_image_path)
        mri_data = mri_scan.get_fdata()
        
        # Get and display basic image information
        print(f"Image Dimensions: {mri_scan.shape}")
        print(f"Voxel Dimensions: {mri_scan.header.get_zooms()}")
        print(f"Data Type: {mri_data.dtype}")
        
        # Additional image properties
        print("Image Properties:")
        print(f"Number of Dimensions: {len(mri_scan.shape)}")
        print(f"Total Voxels: {np.prod(mri_scan.shape)}")
        print(f"Memory Usage: {mri_data.nbytes / 1024 / 1024:.2f} MB")
        
        # Check for data validity
        print("Data Validation:")
        print(f"Value Range: [{np.min(mri_data):.2f}, {np.max(mri_data):.2f}]")
        print(f"Mean Intensity: {np.mean(mri_data):.2f}")
        print(f"Standard Deviation: {np.std(mri_data):.2f}")
        
        # Get orientation information
        orientation = nib.aff2axcodes(mri_scan.affine)
        print(f"Image Orientation: {orientation}")
            
        # Image size 
        print("Image Size", mri_scan.shape)
        # Affine array relating array coordinates from the image data array to coordinates in some world coordinate system (same coordinate system is used for all the images)
        print("Image Affine", mri_scan.affine)
        
        axial_slice = mri_data.shape[2] // 2  # Middle slice along the axial plane
        sagittal_slice = mri_data.shape[0] // 2  # Middle slice along the sagittal plane
        coronal_slice = mri_data.shape[1] // 2  # Middle slice along the coronal plane
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display Axial View (Horizontal)
        axes[0].imshow(np.rot90(mri_data[:, :, axial_slice]), cmap="gray")
        axes[0].set_title("Axial View")
        
        # Display Sagittal View (Side)
        axes[1].imshow(np.rot90(mri_data[sagittal_slice, :, :]), cmap="gray")
        axes[1].set_title("Sagittal View")

        # Display Coronal View (Front)
        axes[2].imshow(np.rot90(mri_data[:, coronal_slice, :]), cmap="gray")
        axes[2].set_title("Coronal View")

        # Remove axis labels
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()  # Added to improve spacing
        plt.show()
        
        return mri_data, mri_scan
        
    def extract_brain_part(self, input_image_path, output_path):
        """Performs skull stripping using FSL BET with improved error handling.
        
        Args:   
           input_image_path = Path to the input MRI scans from which the non brain part has to be removed
           output_path = Path where the extracted part and the new image formed will be saved
           
        Raises:
            RuntimeError: The process failed
           
        """
        
        # basename takes only the file name not the directories before that
        output_path = os.path.join(self.output, output_path)
        
        bet_cmd = f"./app/fsl/bet {input_image_path} {output_path} -R"
        process_run = subprocess.run(bet_cmd, shell=True, capture_output=True, text=True)
        
        # Check if the process is successfully performed or not
        if process_run.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"BET failed!\nError: {process_run.stderr}\nCommand: {bet_cmd}")
        else:
            print("Extracted Brain Part", {output_path})
            return output_path
        
    def intensity_normalisation(self, image_data):
        """Why to normalise the data? 
            Normalisation of the mri intensity is done for consistent and accurate analysis of brain images.
            Reasons : Reduce Scanner and Acquisition Differences
                      Improve Comparability Between Subjects
                      
            Args:
                image_data: input image data (can get it from nib_image.get_fdata() function)
        """
        normalised_image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        return normalised_image_data
    
    def fsl_bias_correction(self, input_image_path, output_path):
        """Bias correction is important because it improves tissue segmentation. Without bias field correction, 
        WM (white matter) may appear as GM (grey matter), or GM as CSF, leading to incorrect segmentation. Bias field 
        differences distort intensity-based alignment when registering MRI scans to a standard atlas. Correction 
        ensures that the same brain structures have similar intensity across subjects. Correction makes changes 
        biologically meaningful rather than scanner-induced.

        Args:
            file_path (str): input file path (brain extracted images)
            output_path (str): output file path (path where new corrected image needs to be saved)

        Raises:
            FileNotFoundError: the path for output file is not correct or the directory does not exists.
        """
        
        fast = fsl.FAST()
        fast.inputs.in_files = input_image_path
        fast.inputs.bias_iters = 5 # how many iterations the algorithm will run to estimate and correct the bias field. (default = 4)
        fast.inputs.bias_lowpass = 20 # Controls the spatial smoothness of the estimated bias field. (default = 10)
        fast.inputs.output_biascorrected = True
        fast.inputs.output_type = 'NIFTI_GZ'
        result = fast.run()
        corrected_file = input_image_path.replace('.nii.gz', '.nii.gz')
        if not os.path.exists(corrected_file):
            raise FileNotFoundError(f"Expected bias-corrected file not found: {corrected_file}")
        os.rename(corrected_file, output_path)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Bias field corrected file not found: {output_path}")
        
        return output_path
        
    def image_registration_mni(self, moving_image_path, template_mni_path, output_prefix):
        try:
        # Read images using correct ANTs syntax
            fixed = ants.image_read(template_mni_path)  # Use from_nibabel instead of image_read
            moving = ants.image_read(moving_image_path)
            
            # Perform registration
            registration = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='SyN', # 'SyN' (Symmetric Normalization) is a non-linear transformation that accounts for complex deformations.
                grad_step=0.2, # Controls the step size for the gradient descent optimization during transformation estimation.
                flow_sigma=3, # Defines the Gaussian smoothing applied to the deformation field during optimization.
                total_sigma=0, # Controls the regularization of the transformation by applying Gaussian smoothing to the total deformation field. (0 = no additional smoothing)
                aff_metric='mattes', # Specifies the similarity metric used for affine transformation estimation.(mattes = Mattes Mutual Information, suitable for multi-modal image registration)
                syn_metric='mattes', # the similarity metric used for the SyN non-linear transformation. ('mattes' is often used for multi-modal images)
            )
            
            # Save warped image
            registered_image_path = os.path.join(output_prefix, 'registered.nii.gz')
            warped_img = registration['warpedmovout'] # warpedmovout = Moving image warped to space of fixed image.
            ants.image_write(warped_img, registered_image_path)
            print(registered_image_path)
            return registered_image_path
            
        except Exception as e:
            print(f"Error during registration: {str(e)}")
            return None
        
class BatchProccessingImages(MRIScansPipeline):
    def __init__(self, input_dir, output_dir, registered_dir):
        super().__init__(input_dir, output_dir)
        self.registered = registered_dir
        
    def process_mri_image(self, args):
        """
        Processes a single MRI image, applying brain extraction, intensity normalization, bias field removal, and registration to MNI space.
        
        Parameters:
        - patient_id (str): Patient identifier.
        - image_id (str): MRI image filename.
        - mni_path (str): Path to MNI template for registration.

        Returns:
        - str: Processed image path or error message.
        """
        patient_id, image_id, mni_path = args
        try:
            patient_path = os.path.join(self.input, patient_id)
            image_path = os.path.join(patient_path, image_id)
            
            # Skip if file doesn't exist
            if not os.path.isfile(image_path):
                return f"File not found: {image_path}"
            
            registered_image_prefix = os.path.join(self.output, f"registered2/{patient_id}-{image_id}-")
            registered_image_path = registered_image_prefix + "registered.nii.gz"
            
            # Skip processing if already registered
            if os.path.exists(registered_image_path):
                return f"Already exists: {registered_image_path}"
            
            # 1. Brain_extraction
            brain_extracted_filename = f"{patient_id}-{image_id}-brain_extracted.nii.gz"
            brain_extracted_path = self.extract_brain_part(image_path, brain_extracted_filename)

            # 2. Intensity Normalisation 
            brain_extracted_img = nib.load(brain_extracted_path)
            brain_extracted_data = brain_extracted_img.get_fdata()
            normalised_image_data = self.intensity_normalisation(brain_extracted_data)
            
            normalised_image = nib.Nifti1Image(normalised_image_data, brain_extracted_img.affine)
            normalised_image_path = os.path.join(self.output, f"{patient_id}-{image_id}-normalized.nii.gz")
            nib.save(normalised_image, normalised_image_path)
            
            # 3. Bias Field Removal
            bias_path = os.path.join(self.output, f"{patient_id}-{image_id}-bias.nii.gz")
            bias_path = self.fsl_bias_correction(normalised_image_path, bias_path)
            
            # 4. MNI Image Registration
            registered_image_path = self.image_registration_mni(bias_path, mni_path, registered_image_prefix)
            return f"Processing complete: {patient_id} - {image_id}, {registered_image_path}"

            
        except Exception as e:
            return f"Error processing {patient_id}-{image_id}: {str(e)}"
        
    def batch_process_mri_images(self, mni_path, num_workers=4):
        """
        Batch processes MRI images in parallel using multiple CPU cores.
        
        Parameters:
        - mni_path (str): Path to the MNI template image for registration.
        - num_workers (int): Number of parallel processes (default: 4).

        Returns:
        - None
        """
        
        tasks = []
        
        # Collect all processing tasks
        for patient_id in os.listdir(self.input):
            patient_path = os.path.join(self.input, patient_id)
            
            # Skip if not a directory
            if not os.path.isdir(patient_path) or patient_id == '.DS_Store':
                continue
            # Get list of MRI images for the patient
            patient_images = [file for file in os.listdir(patient_path) if file.endswith('.nii.gz')]
            
            if not patient_images:
                print(f"No MRI images found for {patient_id}")
                continue
            
            for image_id in patient_images:
                tasks.append((patient_id, image_id, mni_path))
                
        # Run processing in parallel using multiple CPU cores
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(self.process_mri_image, tasks)
            
        for result in results:
            print(results)
        
        print("Batch processing completed successfully!")  
        
class ImageSegmenter(BatchProccessingImages):
    def __init__(self, images_dir, atlas_paths, output_dir):
        """
        Initialize the ImageSegmenter with paths and label mappings.
        
        Args:
            images_dir (str): Directory containing input images
            hoa_path (str): Path to HOA template
            output_dir (str): Base output directory
        """
        
        labels_subcortex = {
            "left_thalamus": 4,
            "right_thalamus": 15,
            "left_hippocampus": 9,
            "right_hippocampus": 19,
            "left_amygdala": 10,
            "right_amygdala": 20,
            "left_cortex": 2,
            "right_cortex": 13,
            "left_lateral_ventricle": 3,
            "right_lateral_ventricle": 14,
            "left_caudate": 5,
            "right_caudate": 16,
            "left_putamen": 6,
            "right_putamen": 17
        }
        
        labels_cortex = {
            "frontal_pole": 1,
            "temporal_pole": 8,
            "superior_frontal_gyrus": 3,
            "middle_frontal_gyrus": 4,
            "superior_parietal_lobule": 18,
            "frontal_medial_cortex": 25,
            "parahippocampal_gyrus_anterior": 34,
            "parahippocampal_gyrus_posterior": 35,
            "occipital_pole": 48,
            "angular_gyrus": 21,
            "frontal_orbital_cortex": 33,
            "lateral_occipital_cortex_inferior": 23,
            "lateral_occipital_cortex_superior": 22,
        }    
        
        mni_labels= {
            "frontal_lobe": 3,
            "occipital_lobe": 5,
            "parietal_lobe": 6,
            "temporal_lobe": 8
        }
        self.images_dir = images_dir
        self.atlas_paths = atlas_paths
        self.output_dir = output_dir
        
        # Define label mappings
        self.label_mappings = {
            'cortex': labels_cortex,
            'subcortex': labels_subcortex,
            'mni': mni_labels
        }
        
        # Create output directories
        for seg_type in ['cortex', 'subcortex', 'mni']:
            os.makedirs(f"{output_dir}/segmented_{seg_type}", exist_ok=True)

    def process_single_image(self, args):
        """
        Process a single image for segmentation.
        
        Args:
            args (tuple): (file_path, segmentation_type)
        
        Returns:
            str: Status message
        """
        file, segmentation_type = args
        
        try:
            if not file.endswith('.nii.gz'):
                return f"Skipped non-nii.gz file: {file}"
                
            registered_mri_path = os.path.join(self.images_dir, file)

            
            if registered_mri_path == "../Output/registered/.DS_Store":
                return f"Skipped .DS_Store file: {registered_mri_path}"
                
            # Load images
            registered_mri_image = nib.load(registered_mri_path)
            registered_mri_data = registered_mri_image.get_fdata()
            
            # Process filename
            # renamed_file_name = file.replace('_registered.nii.gz', '')
            
            # Determine which segmentations to perform
            segmentations = ['cortex', 'subcortex', 'mni'] if segmentation_type == 'all' else [segmentation_type]
            
            # Process each segmentation type
            for seg_type in segmentations:
                atlas_path = self.atlas_paths[seg_type]
                atlas_img = nib.load(atlas_path)
                atlas_data = atlas_img.get_fdata()
                
                labels = self.label_mappings[seg_type]
                labels = self.label_mappings[seg_type]
                
                for label in labels:
                    label_out_path = f"{self.output_dir}/segmented_{seg_type}/{label}.nii.gz"
                    
                    # Skip if output already exists
                    if os.path.exists(label_out_path):
                        print("Already exists", seg_type)
                        continue
                    
                    # Create and apply mask
                    label_mask = np.isin(atlas_data, labels[label]).astype(np.float32)
                    label_mask_img = nib.Nifti1Image(label_mask, atlas_img.affine)
                    
                    resampled_mask_img = resample_from_to(label_mask_img, registered_mri_image, order=0)
                    label_data = registered_mri_data * resampled_mask_img.get_fdata()
                    label_img = nib.Nifti1Image(label_data, registered_mri_image.affine)
                    
                    nib.save(label_img, label_out_path)
            
            return f"Successfully processed {file}"
            
        except Exception as e:
            return f"Error processing {file}: {str(e)}"

    def batch_segment_images(self, segmentation_type='all', num_workers=4):
        """
        Batch process images for segmentation using parallel processing.
        
        Args:
            segmentation_type (str): Type of segmentation ('cortex', 'subcortex', 'mni', or 'all')
            num_workers (int): Number of parallel processes to use
        """
        # Collect all processing tasks
        tasks = []
        files = [f for f in os.listdir(self.images_dir) if f.endswith('.nii.gz')]
        
        for file in files:
            
            tasks.append((file, segmentation_type))
        
        if not tasks:
            print("No NIFTI files found in the input directory")
            return
        
        print(f"Starting batch processing of {len(tasks)} files with {num_workers} workers...")
        
        # Process tasks in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.process_single_image, tasks))
        
        # Print results
        for result in results:
            print(result)
        
        print("Batch processing completed!")
        
    def plot_segmented_img(self, bg_img_path, segmented_img_path):
        """Plot any of the required segmented brain part on top of the entire brain image

        Args:
            bg_img_path (str): path of the entire brain image
            segmented_img_path (str): path of the segmented brain part image 
        """
        
        # Load the entire brain MRI image
        bg_mri = nib.load(bg_img_path)
        bg_data = bg_mri.get_fdata()
        
        # Load the segmented brain part image
        segmented_mri = nib.load(segmented_img_path)
        segmented_data = segmented_mri.get_fdata()
        
        # Find slices with maximum segmentation in each view
        axial_slice = np.argmax(np.sum(segmented_data, axis=(0, 1)))    # Z-axis
        sagittal_slice = np.argmax(np.sum(segmented_data, axis=(1, 2)))  # X-axis
        coronal_slice = np.argmax(np.sum(segmented_data, axis=(0, 2)))   # Y-axis
        
        # Define custom colormap blending 'cool' and 'hot'
        colors = ["cyan", "blue", "red", "yellow"]
        custom_cmap = LinearSegmentedColormap.from_list("hotcool", colors, N=256)

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{segmented_img_path.split('-')[-1]}".replace(".nii.gz", ""))

        # Mask segmentation data to avoid coloring the background
        seg_data_masked = np.ma.masked_where(segmented_data == 0, segmented_data)

        # Axial View
        axes[0].imshow(bg_data[:, :, axial_slice], cmap='gray', alpha=1)
        axes[0].imshow(seg_data_masked[:, :, axial_slice], cmap=custom_cmap, alpha=0.7)
        axes[0].set_title(f"Axial View (Slice {axial_slice})")

        # Sagittal View (Rotate 180°)
        axes[1].imshow(np.rot90(bg_data[sagittal_slice, :, :].T, 2), cmap='gray', alpha=1)
        axes[1].imshow(np.rot90(seg_data_masked[sagittal_slice, :, :].T, 2), cmap=custom_cmap, alpha=0.7)
        axes[1].set_title(f"Sagittal View (Slice {sagittal_slice})")

        # Coronal View (Rotate 180°)
        axes[2].imshow(np.rot90(bg_data[:, coronal_slice, :].T, 2), cmap='gray', alpha=1)
        axes[2].imshow(np.rot90(seg_data_masked[:, coronal_slice, :].T, 2), cmap=custom_cmap, alpha=0.7)
        axes[2].set_title(f"Coronal View (Slice {coronal_slice})")


        plt.tight_layout()
        plt.show()

class FeatureExtraction(ImageSegmenter):
    def __init__(self, images_dir, atlas_paths, output_dir):
        super().__init__(images_dir, atlas_paths, output_dir)
        
    def extract_volume(self, segmented_img_path):
        """
        Calculate volume using integral image approach for faster computation.
        Alternative to simple voxel counting.
        """
        img = nib.load(segmented_img_path)
        data = img.get_fdata()
        voxel_size = np.prod(img.header.get_zooms())
        
        # Create integral image
        integral_img = np.cumsum(np.cumsum(np.cumsum(data, axis=0), axis=1), axis=2)
        volume = integral_img[-1, -1, -1] * voxel_size
        
        return volume
    
    def extract_surface_area(self, segmented_image_path, method="greyscale", percentile=50):
        """
        Calculate surface area using gradient-based approach.
        Alternative to marching cubes.
        """
        img = nib.load(segmented_image_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()
        
        if method == 'binary':
            # Convert to binary using Otsu's method or a threshold of 0.5
            threshold = filters.threshold_otsu(data)
            data_processed = data > threshold
            level = 0  # Use 0 for binary data
        else:
            # Use the data as is
            data_processed = data
            # Set level to mean between min and max of non-zero values
            non_zero_data = data[data > 0]
            level = np.percentile(non_zero_data, percentile) if len(non_zero_data) > 0 else 0
        
        # Calculate surface area
        verts, faces, _, _ = measure.marching_cubes(data_processed, level=level)
        verts = verts * spacing
        return measure.mesh_surface_area(verts, faces)
    
    def extract_compactness(self, volume, surface_area):
        """Calculate compactness using surface area cubed over volume squared."""
        if volume == 0:
            return 0
        return (surface_area)**3/(volume)**2
    
    def extract_sphericity(self, volume, surface_area):
        """Calculate sphericity using the surface area/volume approach."""
        if surface_area == 0:
            return 0
        return (np.pi**(1/3)) * (6 * volume)**(2/3) / surface_area
    
    def extract_eccentricity(self, segmented_image_path):
        """
        Calculate eccentricity using improved ellipsoid fitting approach.
        
        Parameters:
        segmented_image_path (str): Path to the segmented image file
        
        Returns:
        float: Calculated eccentricity value
        """
        # Load the image
        img = nib.load(segmented_image_path)
        data = img.get_fdata()
        spacing = img.header.get_zooms()
        
        # Get coordinates of object points with proper spacing
        coords = np.array(np.where(data > 0)).T * spacing
        
        if len(coords) < 4:
            print(f"Warning: Not enough points for ellipsoid fitting in {segmented_image_path}")
            return 0
        
        # Center the coordinates
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid
        
        try:
            # Use SVD instead of eigenvalue decomposition for better numerical stability
            U, S, Vh = svd(coords_centered, full_matrices=False)
            
            # The singular values are the square roots of the eigenvalues
            # of the covariance matrix
            eigenvalues = S ** 2
            
            # Sort eigenvalues in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Add small epsilon to prevent division by zero
            epsilon = 1e-10
            eigenvalues = eigenvalues + epsilon
            
            # Calculate eccentricity using the classical formula
            # e = sqrt(1 - (minor_axis^2)/(major_axis^2))
            eccentricity = np.sqrt(1 - eigenvalues[2] / eigenvalues[0])
            
            # Print diagnostic information
            # print(f"\nDiagnostic information for {segmented_image_path}:")
            # print(f"Number of points: {len(coords)}")
            # print(f"Eigenvalues: {eigenvalues}")
            # print(f"Calculated eccentricity: {eccentricity}")
            
            # Validate the eccentricity value
            if not (0 <= eccentricity <= 1):
                print(f"Warning: Invalid eccentricity value {eccentricity}")
                return 0
                
            # Optional: Visualization
            # if False:  # Set to True for debugging
            #     visualize_ellipsoid(coords, eigenvalues, Vh, centroid, segmented_image_path)
                
            return eccentricity
            
        except Exception as e:
            print(f"Error calculating eccentricity for {segmented_image_path}: {str(e)}")
            return 0
    
    def extract_features(self, segmented_dir, output_csv=None):
        files = os.listdir(segmented_dir)

        # Initialize a dictionary to store features for each MRI_ID
        features_dict = {}

        # Process each file in the directory
        for file in files:
            if not file.endswith('.nii.gz'):
                continue
            
            # Extract the part name from the filename
            part_name = file.replace('.nii.gz', '')
            
            # Construct the full file path
            image_path = os.path.join(segmented_dir, file)
            
            if os.path.exists(image_path):
                try:
                    # Extract volume
                    volume = self.extract_volume(image_path)
                    
                    # Extract surface area
                    surface_area = self.extract_surface_area(image_path)
                    
                    # Extract shape descriptors
                    compactness = self.extract_compactness(volume, surface_area)
                    sphericity = self.extract_sphericity(volume, surface_area)
                    eccentricity = self.extract_eccentricity(image_path)
                    
                    # If the MRI_ID is not yet in the dictionary, add it
                    mri_id = 0
                    if mri_id not in features_dict:
                        features_dict[mri_id] = {}

                    # Add the features for this brain part to the dictionary
                    features_dict[mri_id][f"{part_name}_Volume"] = float(volume)
                    features_dict[mri_id][f"{part_name}_Surface_Area"] = float(surface_area)
                    features_dict[mri_id][f"{part_name}_Compactness"] = float(compactness)
                    features_dict[mri_id][f"{part_name}_Sphericity"] = float(sphericity)
                    features_dict[mri_id][f"{part_name}_Eccentricity"] = float(eccentricity)
                
                except Exception as e:
                    # Print the file path and error if an error occurs
                    print(f"Error processing file {image_path}: {e}")
                    continue  # Skip to the next file

        # If output_csv is provided, save to CSV
        if output_csv:
            # Convert the dictionary to a DataFrame for CSV export
            features_df = pd.DataFrame.from_dict(features_dict, orient='index').reset_index()
            features_df.rename(columns={'index': 'MRI_ID'}, inplace=True)
            features_df.to_csv(output_csv, index=False)
            print("Features extracted and saved to:", output_csv)

        # Convert nested dictionary to JSON-compatible format
        json_features = {}
        for mri_id, features in features_dict.items():
            # Convert numeric ID to string key for JSON
            json_features[str(mri_id)] = features
        
        # Return the features as a JSON-compatible dictionary
        return json_features
        
        

if __name__ == "__main__":
    
    # Labels for
    labels_subcortex = {
        "left_thalamus": 4,
        "right_thalamus": 15,
        "left_hippocampus": 9,
        "right_hippocampus": 19,
        "left_amygdala": 10,
        "right_amygdala": 20,
        "left_cortex": 2,
        "right_cortex": 13,
        "left_lateral_ventricle": 3,
        "right_lateral_ventricle": 14,
        "left_caudate": 5,
        "right_caudate": 16,
        "left_putamen": 6,
        "right_putamen": 17
    }
    
    labels_cortex = {
        "frontal_pole": 1,
        "temporal_pole": 8,
        "superior_frontal_gyrus": 3,
        "middle_frontal_gyrus": 4,
        "superior_parietal_lobule": 18,
        "frontal_medial_cortex": 25,
        "parahippocampal_gyrus_anterior": 34,
        "parahippocampal_gyrus_posterior": 35,
        "occipital_pole": 48,
        "angular_gyrus": 21,
        "frontal_orbital_cortex": 33,
        "lateral_occipital_cortex_inferior": 23,
        "lateral_occipital_cortex_superior": 22,
    }    
    
    mni_labels= {
        "frontal_lobe": 3,
        "occipital_lobe": 5,
        "parietal_lobe": 6,
        "temporal_lobe": 8
    }        
    
    atlas_paths = {
        'cortex': "../../ll/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
        'subcortex': "../../ll/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
        'mni': "../../ll/data/atlases//MNI/MNI-maxprob-thr25-1mm.nii.gz"
    }
    
    input = '../CogNIDImages'
    output = '../Output'
    registered = '../Output/registered'
  
    mni_path = "../../ll/pkgs/fsl-data_standard-2208.0-0/data/standard/MNI152_T1_1mm_brain.nii.gz"
    segmenter = ImageSegmenter(registered, atlas_paths, output)
    # segmenter.batch_segment_images(num_workers=8) 
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-right_hippocampus.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-left_hippocampus.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-right_caudate.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-left_caudate.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-left_thalamus.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID221-CogNID221-right_thalamus.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-left_amygdala.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-right_amygdala.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID221-CogNID221-left_cortex.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID221-CogNID221-right_cortex.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-left_lateral_ventricle.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-right_lateral_ventricle.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-left_putamen.nii.gz')
    segmenter.plot_segmented_img('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz', '../Output/segmented_subcortex/CogNID011-CogNID011-right_putamen.nii.gz')
    
    # segmenter.display_mri_image('../Output/registered/CogNID011-CogNID011.nii.gz-_warped.nii.gz')