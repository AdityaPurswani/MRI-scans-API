import shutil
import os
from MRIScansPipeline import MRIScansPipeline, ImageSegmenter, FeatureExtraction
import nibabel as nib


class MRIProcessingPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pipeline = MRIScansPipeline("uploads", "processed")
        self.segmented_dir = "processed/segmented"

    def clean_segmented_directory(self):
        """ Removes old segmentation results before processing a new MRI scan. """
        if os.path.exists(self.segmented_dir):
            print("🔹 Clearing previous segmentation results...")
            shutil.rmtree(self.segmented_dir)  # Delete all existing segmented files
        os.makedirs(self.segmented_dir, exist_ok=True)  # Recreate empty directory

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

        print("🔹 Starting processing...")

        # Step 1: Extract Brain
        print(f"➡ Extracting brain to {brain_path}")
        self.pipeline.extract_brain_part(self.file_path, brain_path)

        # Step 2: Intensity Normalization
        print(f"➡ Normalizing intensity to {norm_path}")
        brain_image = nib.load(brain_path)
        brain_data = brain_image.get_fdata()
        norm_data = self.pipeline.intensity_normalisation(brain_data)
        nib.save(nib.Nifti1Image(norm_data, brain_image.affine), norm_path)

        # Step 3: Bias Field Correction
        print(f"➡ Applying bias correction to {bias_corrected_path}")
        self.pipeline.fsl_bias_correction(norm_path, bias_corrected_path)

        # Step 4: Image Registration
        mni_template = "/Users/adityapurswani/Documents/ll/pkgs/fsl-data_standard-2208.0-0/data/standard/MNI152_T1_1mm_brain.nii.gz"
        print(f"➡ Registering to MNI space at {registered_path}")
        self.pipeline.image_registration_mni(bias_corrected_path, mni_template, "processed/")

        # **Step 5: Clear Segmentation Directory Before Running New Segmentation**
        self.clean_segmented_directory()

        print(f"➡ Segmenting brain images in {input_dir} → Output: {output_dir}")
        segmenter = ImageSegmenter(input_dir, atlas_paths, output_dir)

        # Process specific images
        specific_files = ["registered.nii.gz"]
        for file in specific_files:
            segmenter.process_single_image((file, 'all'))

        # Step 6: Extract Volumetric Features
        print(f"➡ Extracting volumetric features in {output_dir} → Output: {feature_output_dir}")
        feature_extractor = FeatureExtraction(output_dir, atlas_paths, feature_output_dir)

        segmented_cortex = feature_extractor.extract_features(f"{output_dir}/segmented_cortex", "processed/volumetrics_cortex.csv")
        segmented_subcortex = feature_extractor.extract_features(f"{output_dir}/segmented_subcortex", "processed/volumetrics_subcortex.csv")
        segmented_mni = feature_extractor.extract_features(f"{output_dir}/segmented_mni", "processed/volumetrics_mni.csv")

        print("✅ Processing complete")
        print("🔹 Cortex Volume Data:", segmented_cortex)
        print("🔹 Subcortex Volume Data:", segmented_subcortex)
        print("🔹 MNI Volume Data:", segmented_mni)

        return {
            "processed_files": {
                "brain_extracted": brain_path,
                "normalized": norm_path,
                "bias_corrected": bias_corrected_path,
                "registered": registered_path
            },
            "segmented_cortex": segmented_cortex,
            "segmented_subcortex": segmented_subcortex,
            "segmented_mni": segmented_mni
        }
