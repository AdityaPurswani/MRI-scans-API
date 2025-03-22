import shutil
import os
from MRIScansPipeline import MRIScansPipeline, ImageSegmenter, FeatureExtraction
import nibabel as nib


class MRIProcessingPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_dir = "/Users/adityapurswani/Documents/MRIscansAPI"
        self.pipeline = MRIScansPipeline("uploads", "processed")
        self.segmented_dir = os.path.join(self.base_dir, "processed/segmented")
        # Add a new directory for uncompressed files
        self.uncompressed_dir = os.path.join(self.base_dir, "processed/uncompressed")

    def clean_segmented_directory(self):
        """ Removes old segmentation results before processing a new MRI scan. """
        if os.path.exists(self.segmented_dir):
            print("Clearing previous segmentation results...")
            shutil.rmtree(self.segmented_dir)  # Delete all existing segmented files
        os.makedirs(self.segmented_dir, exist_ok=True)  # Recreate empty directory
        
    def clean_uncompressed_directory(self):
        """ Removes old uncompressed files before processing a new MRI scan. """
        if os.path.exists(self.uncompressed_dir):
            print("Clearing previous uncompressed files...")
            shutil.rmtree(self.uncompressed_dir)  # Delete all existing uncompressed files
        os.makedirs(self.uncompressed_dir, exist_ok=True)  # Recreate empty directory

    def save_uncompressed_copy(self, img_path, output_filename):
        """ Save an uncompressed copy of a NIfTI file """
        try:
            # Load the compressed image
            img = nib.load(img_path)
            # Create the uncompressed filename (remove .gz extension if present)
            uncompressed_path = os.path.join("/Users/adityapurswani/Documents/MRIscansAPI", self.uncompressed_dir, output_filename)
            if uncompressed_path.endswith('.gz'):
                uncompressed_path = uncompressed_path[:-3]
            # Save as uncompressed NIfTI
            # Ensure directory exists
            os.makedirs(os.path.dirname(uncompressed_path), exist_ok=True)
            nib.save(img, uncompressed_path)
            print(f"Saved uncompressed copy at: {uncompressed_path}")
            return uncompressed_path
        except Exception as e:
            print(f"⚠️ Error saving uncompressed copy: {e}")
            return None

    def run_pipeline(self):
        """ Runs full MRI processing and clears old segmentation before running new segmentation """

        brain_path = "/Users/adityapurswani/Documents/MRIscansAPI/processed/brain.nii.gz"
        norm_path = "/Users/adityapurswani/Documents/MRIscansAPI/processed/normalized.nii.gz"
        bias_corrected_path = "/Users/adityapurswani/Documents/MRIscansAPI/processed/bias_corrected.nii.gz"
        registered_path = "/Users/adityapurswani/Documents/MRIscansAPI/processed/registered.nii.gz"

        input_dir = "/Users/adityapurswani/Documents/MRIscansAPI/processed"
        output_dir = "/Users/adityapurswani/Documents/MRIscansAPI/processed/segmented"
        feature_output_dir = "/Users/adityapurswani/Documents/MRIscansAPI/processed/features"

        atlas_paths = {
            'cortex': "/Users/adityapurswani/Documents/ll/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz",
            'subcortex': "/Users/adityapurswani/Documents/ll/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
            'mni': "/Users/adityapurswani/Documents/ll/data/atlases/MNI/MNI-maxprob-thr25-1mm.nii.gz"
        }

        print("Starting processing...")
        
        # Initialize uncompressed directory
        self.clean_uncompressed_directory()
        
        # Create uncompressed copy of input file
        input_filename = os.path.basename(self.file_path)
        input_uncompressed_path = self.save_uncompressed_copy(self.file_path, input_filename)

        # Step 1: Extract Brain
        print(f"➡ Extracting brain to {brain_path}")
        self.pipeline.extract_brain_part(self.file_path, brain_path)
        # Save uncompressed copy
        brain_uncompressed_path = self.save_uncompressed_copy(brain_path, "brain.nii")

        # Step 2: Intensity Normalization
        print(f"➡ Normalizing intensity to {norm_path}")
        brain_image = nib.load(brain_path)
        brain_data = brain_image.get_fdata()
        norm_data = self.pipeline.intensity_normalisation(brain_data)
        nib.save(nib.Nifti1Image(norm_data, brain_image.affine), norm_path)
        # Save uncompressed copy
        norm_uncompressed_path = self.save_uncompressed_copy(norm_path, "normalized.nii")

        # Step 3: Bias Field Correction
        print(f"➡ Applying bias correction to {bias_corrected_path}")
        self.pipeline.fsl_bias_correction(norm_path, bias_corrected_path)
        # Save uncompressed copy
        bias_corrected_uncompressed_path = self.save_uncompressed_copy(bias_corrected_path, "bias_corrected.nii")

        # Step 4: Image Registration
        mni_template = "/Users/adityapurswani/Documents/ll/pkgs/fsl-data_standard-2208.0-0/data/standard/MNI152_T1_1mm_brain.nii.gz"
        print(f"➡ Registering to MNI space at {registered_path}")
        self.pipeline.image_registration_mni(bias_corrected_path, mni_template, "processed/")
        # Save uncompressed copy
        registered_uncompressed_path = self.save_uncompressed_copy(registered_path, "registered.nii")

        # **Step 5: Clear Segmentation Directory Before Running New Segmentation**
        self.clean_segmented_directory()

        print(f"➡ Segmenting brain images in {input_dir} → Output: {output_dir}")
        segmenter = ImageSegmenter(input_dir, atlas_paths, output_dir)

        # Process specific images
        specific_files = ["registered.nii.gz"]
        for file in specific_files:
            segmenter.process_single_image((file, 'all'))
            
                    # Save uncompressed copies of segmentation results
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.nii.gz'):
                    # Create relative path to maintain directory structure
                    rel_path = os.path.relpath(root, output_dir)
                    # Create same directory structure in uncompressed folder
                    uncompressed_subdir = os.path.join(self.uncompressed_dir, rel_path)
                    os.makedirs(uncompressed_subdir, exist_ok=True)
                    # Calculate source path
                    source_path = os.path.join(root, file)
                    # Calculate output path
                    output_rel_path = os.path.join(rel_path, file[:-3])
                    self.save_uncompressed_copy(source_path, output_rel_path)

        # Step 6: Extract Volumetric Features
        print(f"➡ Extracting volumetric features in {output_dir} → Output: {feature_output_dir}")
        feature_extractor = FeatureExtraction(output_dir, atlas_paths, feature_output_dir)

        segmented_cortex = feature_extractor.extract_features(f"{output_dir}/segmented_cortex", "processed/volumetrics_cortex.csv")
        segmented_subcortex = feature_extractor.extract_features(f"{output_dir}/segmented_subcortex", "processed/volumetrics_subcortex.csv")
        segmented_mni = feature_extractor.extract_features(f"{output_dir}/segmented_mni", "processed/volumetrics_mni.csv")

        print("Processing complete")
        print("Cortex Volume Data:", segmented_cortex)
        print("Subcortex Volume Data:", segmented_subcortex)
        print("MNI Volume Data:", segmented_mni)

        return {
            "input_file": {
                "compressed": self.file_path,
                "uncompressed": input_uncompressed_path
            },
            "uncompressed_files": {
                "brain_extracted": brain_uncompressed_path,
                "normalized": norm_uncompressed_path,
                "bias_corrected": bias_corrected_uncompressed_path,
                "registered": registered_uncompressed_path
            },
            "segmented_cortex": segmented_cortex,
            "segmented_subcortex": segmented_subcortex,
            "segmented_mni": segmented_mni
        }