import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils
import pandas as pd
from collections import Counter
from .medical_augmentation import create_augmentation_pipeline


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False, csv_path=None, cls_head=False, 
                 use_augmentation=False, augmentation_config='default'):
        '''
        Enhanced BRATS Dataset with pathology grade classification support and data augmentation
        
        Args:
            directory: Path to BRATS dataset directory
            transform: Data transformation pipeline
            test_flag: Whether this is for testing (no segmentation labels)
            csv_path: Path to name_mapping.csv file containing pathology grades
            cls_head: Whether to include pathology grade labels for classification
            use_augmentation: Whether to use data augmentation
            augmentation_config: Augmentation configuration ('default', 'light', 'heavy')
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag
        self.cls_head = cls_head
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config
        
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        
        # Load pathology grade mapping if CSV path is provided
        self.grade_mapping = {}
        self.class_weights = None
        if csv_path and os.path.exists(csv_path):
            self.grade_mapping = self._load_grade_mapping(csv_path)
            if self.cls_head:
                self.class_weights = self._calculate_class_weights()
                print(f"Class weights calculated: {self.class_weights}")
        
        # Initialize data augmentation pipeline
        if self.use_augmentation:
            self.augmentation_pipeline = create_augmentation_pipeline(
                is_training=not test_flag, 
                config_name=self.augmentation_config
            )
            print(f"Data augmentation enabled with config: {self.augmentation_config}")
        else:
            self.augmentation_pipeline = None
        
        # Build database
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
    
    def _load_grade_mapping(self, csv_path):
        """Load pathology grade mapping from CSV file"""
        grade_mapping = {}
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with {len(df)} entries")
            
            for _, row in df.iterrows():
                subject_id = row['BraTS_2020_subject_ID']
                grade = row['Grade']  # HGG or LGG
                # Convert to binary: HGG=1, LGG=0
                grade_mapping[subject_id] = 1 if grade == 'HGG' else 0
            
            print(f"Grade mapping loaded: {len(grade_mapping)} subjects")
            grade_counts = Counter(grade_mapping.values())
            print(f"Grade distribution: LGG={grade_counts[0]}, HGG={grade_counts[1]}")
            
        except Exception as e:
            print(f"Error loading grade mapping: {e}")
            print("Continuing without pathology grade labels...")
            
        return grade_mapping
    
    def _calculate_class_weights(self):
        """Calculate class weights to handle class imbalance"""
        if not self.grade_mapping:
            return None
            
        grade_counts = Counter(self.grade_mapping.values())
        total_samples = sum(grade_counts.values())
        
        # Calculate weights using inverse frequency
        weight_lgg = total_samples / (2 * grade_counts[0]) if grade_counts[0] > 0 else 1.0
        weight_hgg = total_samples / (2 * grade_counts[1]) if grade_counts[1] > 0 else 1.0
        
        weights = torch.tensor([weight_lgg, weight_hgg], dtype=torch.float32)
        print(f"Class weights: LGG={weight_lgg:.3f}, HGG={weight_hgg:.3f}")
        
        return weights
    
    def _extract_subject_id(self, file_path):
        """Extract subject ID from file path"""
        # Extract subject ID from path like: .../BraTS20_Training_001/BraTS20_Training_001_t1.nii
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if 'BraTS20_Training_' in part:
                return part
        return None

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path = filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        
        # Extract subject ID for pathology grade lookup
        subject_id = self._extract_subject_id(path)
        grade_label = 0  # Default to LGG
        if self.cls_head and subject_id in self.grade_mapping:
            grade_label = self.grade_mapping[subject_id]
        
        if self.test_flag:
            image = out
            image = image[..., 8:-8, 8:-8]     # crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            
            if self.cls_head:
                # 修复：只返回一个image，避免DataLoader错误拼接
                return (image, grade_label, path)
            else:
                # 修复：只返回一个image，避免DataLoader错误拼接
                return (image, path)
        else:
            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      # crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label = torch.where(label > 0, 1, 0).float()  # merge all tumor classes into one
            
            # ==================== Data Augmentation ====================
            # BRATS数据集：多模态医学图像（T1, T1ce, T2, FLAIR）
            # 关键：确保所有模态和掩码使用相同的随机变换参数
            
            # 应用医学图像专用数据增强
            if self.augmentation_pipeline is not None:
                image, label, grade_label = self.augmentation_pipeline(image, label.squeeze(0), grade_label)
                label = label.unsqueeze(0)  # 恢复batch维度
            
            # 应用标准变换
            if self.transform:
                state = torch.get_rng_state()  # 保存随机状态
                image = self.transform(image)  # 对多模态图像应用变换
                torch.set_rng_state(state)    # 恢复随机状态
                label = self.transform(label)  # 对掩码应用相同变换
            
            if self.cls_head:
                return (image, label, grade_label, path)
            else:
                return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False, csv_path=None, use_cls_head=False,
                 use_augmentation=False, augmentation_config='default', split_ratio=0.8, split_mode='train'):
        '''
        Enhanced BRATS 3D Dataset with pathology grade classification support and data augmentation
        
        Args:
            directory: Path to BRATS dataset directory
            transform: Data transformation pipeline
            test_flag: Whether this is for testing (no segmentation labels)
            csv_path: Path to name_mapping.csv file containing pathology grades
            use_cls_head: Whether to include pathology grade labels for classification
            use_augmentation: Whether to use data augmentation
            augmentation_config: Augmentation configuration ('default', 'light', 'heavy')
            split_ratio: Ratio for train/validation split (default: 0.8 for 80% train, 20% validation)
            split_mode: 'train' or 'validation' - which part of the split to use
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag
        self.use_cls_head = use_cls_head
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config
        self.split_ratio = split_ratio
        self.split_mode = split_mode
        
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        
        # Load pathology grade mapping if CSV path is provided
        self.grade_mapping = {}
        self.class_weights = None

        # Class weights calculation
        if csv_path and os.path.exists(csv_path):
            self.grade_mapping = self._load_grade_mapping(csv_path)
            if self.use_cls_head:
                self.class_weights = self._calculate_class_weights()
                print(f"Class weights calculated: {self.class_weights}")
        
        # Initialize data augmentation pipeline
        if self.use_augmentation:
            self.augmentation_pipeline = create_augmentation_pipeline(
                is_training=not test_flag, 
                config_name=self.augmentation_config
            )
            print(f"Data augmentation enabled with config: {self.augmentation_config}")
        else:
            self.augmentation_pipeline = None
        
        # Build database
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    # Handle both .nii and .nii.gz formats
                    if f.endswith('.nii.gz') or f.endswith('.nii'):
                        # Extract sequence type from filename
                        # Expected format: BraTS20_Training_XXX_sequence.nii[.gz]
                        parts = f.split('_')
                        if len(parts) >= 4:
                            # Get the sequence type (t1, t1ce, t2, flair, seg)
                            seqtype = parts[3].split('.')[0]
                            # Only add if it's one of the expected sequence types
                            if seqtype in self.seqtypes_set:
                                datapoint[seqtype] = os.path.join(root, f)
                            else:
                                print(f"Warning: Unknown sequence type '{seqtype}' in file: {f}")
                        else:
                            print(f"Warning: Unexpected file format: {f}")
                            continue
                    else:
                        continue  # Skip non-NIfTI files
                
                # Check if we have all required sequences for this datapoint
                required_seqtypes = ['t1', 't1ce', 't2', 'flair']
                if not test_flag:
                    required_seqtypes.append('seg')
                
                missing_seqtypes = [seq for seq in required_seqtypes if seq not in datapoint]
                
                if not missing_seqtypes:
                    # All required sequences are present
                    self.database.append(datapoint)
                else:
                    print(f"Warning: Skipping incomplete datapoint with missing sequences: {missing_seqtypes}")
        
        # Apply train/validation split if needed
        if hasattr(self, 'split_ratio') and self.split_ratio > 0:
            self._apply_train_val_split()
    
    def _load_grade_mapping(self, csv_path):
        """Load pathology grade mapping from CSV file"""
        grade_mapping = {}
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with {len(df)} entries")
            
            for _, row in df.iterrows():
                subject_id = row['BraTS_2020_subject_ID']
                grade = row['Grade']  # HGG or LGG
                # Convert to binary: HGG=1, LGG=0
                grade_mapping[subject_id] = 1 if grade == 'HGG' else 0
            
            print(f"Grade mapping loaded: {len(grade_mapping)} subjects")
            grade_counts = Counter(grade_mapping.values())
            print(f"Grade distribution: LGG={grade_counts[0]}, HGG={grade_counts[1]}")
            
        except Exception as e:
            print(f"Error loading grade mapping: {e}")
            print("Continuing without pathology grade labels...")
            
        return grade_mapping
    
    def _calculate_class_weights(self):
        """Calculate class weights to handle class imbalance"""
        if not self.grade_mapping:
            return None
            
        grade_counts = Counter(self.grade_mapping.values())
        total_samples = sum(grade_counts.values())
        
        # Calculate weights using inverse frequency
        weight_lgg = total_samples / (2 * grade_counts[0]) if grade_counts[0] > 0 else 1.0
        weight_hgg = total_samples / (2 * grade_counts[1]) if grade_counts[1] > 0 else 1.0
        
        weights = torch.tensor([weight_lgg, weight_hgg], dtype=torch.float32)
        print(f"Class weights: LGG={weight_lgg:.3f}, HGG={weight_hgg:.3f}")
        
        return weights
    
    def _extract_subject_id(self, file_path):
        """Extract subject ID from file path"""
        # Extract subject ID from path like: .../BraTS20_Training_001/BraTS20_Training_001_t1.nii
        path_parts = file_path.split(os.sep)
        for part in path_parts:
            if 'BraTS20_Training_' in part:
                return part
        return None
    
    def __len__(self):
        return len(self.database) * 155

    def __getitem__(self, x):
        out = []
        n = x // 155
        slice_idx = x % 155
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice_idx]
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        
        # Extract subject ID for pathology grade lookup
        subject_id = self._extract_subject_id(path)
        grade_label = 0  # Default to LGG
        if self.use_cls_head and subject_id in self.grade_mapping:
            grade_label = self.grade_mapping[subject_id]
        
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            
            virtual_path = path.split('.nii')[0] + "_slice" + str(slice_idx) + ".nii"
            if self.use_cls_head:
                # 修复：只返回一个image，避免DataLoader错误拼接
                return (image, grade_label, virtual_path)
            else:
                # 修复：只返回一个image，避免DataLoader错误拼接
                return (image, virtual_path)
        else:
            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            
            # ==================== 数据标准化 ====================
            # 对每个模态分别进行标准化到[0, 1]范围
            # 这对于扩散模型的收敛至关重要！
            for i in range(image.shape[0]):
                modal = image[i]
                modal_min = modal.min()
                modal_max = modal.max()
                if modal_max > modal_min:
                    image[i] = (modal - modal_min) / (modal_max - modal_min + 1e-8)
                else:
                    image[i] = torch.zeros_like(modal)
            # =====================================================
            
            if self.transform:
                state = torch.get_rng_state()  # 保存随机状态
                image = self.transform(image)  # 对多模态图像应用变换
                torch.set_rng_state(state)    # 恢复随机状态
                label = self.transform(label)  # 对掩码应用相同变换
            
            virtual_path = path.split('.nii')[0] + "_slice" + str(slice_idx) + ".nii"
            if self.use_cls_head:
                return (image, label, grade_label, virtual_path)
            else:
                return (image, label, virtual_path)
    
    def _apply_train_val_split(self):
        """Apply train/validation split to the database"""
        import random
        
        # Set random seed for reproducible splits
        random.seed(42)
        
        # Get all subject IDs
        subject_ids = []
        for datapoint in self.database:
            # Extract subject ID from any file path
            file_path = list(datapoint.values())[0]
            subject_id = self._extract_subject_id(file_path)
            if subject_id:
                subject_ids.append(subject_id)
        
        # Shuffle subject IDs
        random.shuffle(subject_ids)
        
        # Calculate split sizes
        total_subjects = len(subject_ids)
        train_size = int(total_subjects * self.split_ratio)
        
        if self.split_mode == 'train':
            selected_subjects = set(subject_ids[:train_size])
        else:  # validation
            selected_subjects = set(subject_ids[train_size:])
        
        # Filter database to only include selected subjects
        filtered_database = []
        for datapoint in self.database:
            file_path = list(datapoint.values())[0]
            subject_id = self._extract_subject_id(file_path)
            if subject_id in selected_subjects:
                filtered_database.append(datapoint)
        
        self.database = filtered_database
        print(f"Applied {self.split_mode} split: {len(self.database)} subjects selected from {total_subjects} total")



