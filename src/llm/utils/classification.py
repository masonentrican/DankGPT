"""
Utilities for classification fine tuning tasks.
"""

import pandas as pd

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split a dataset into three parts: train, test, and validation.
    
    Args:
        df: DataFrame containing the dataset
        train_ratio: Ratio of the dataset to be used for training (default: 0.7)
        val_ratio: Ratio of the dataset to be used for validation (default: 0.1)
        random_state: Random seed for reproducible sampling (default: 123)
    
    Returns:
        train_df: DataFrame containing the training data
        val_df: DataFrame containing the validation data
        test_df: DataFrame containing the testing data
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True) # Shuffle the dataset

    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df


def balance_two_class_dataset(
    df: pd.DataFrame,
    label_column: str = "Label",
    class1: str | None = None,
    class2: str | None = None,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Balance a two-class dataset by resampling the minority class to match the majority class size.
    
    Args:
        df: DataFrame containing the dataset
        label_column: Name of the column containing class labels (default: "Label")
        class1: Name of the first class. If None, will be auto-detected as the minority class.
        class2: Name of the second class. If None, will be auto-detected as the majority class.
        random_state: Random seed for reproducible sampling (default: 123)
    
    Returns:
        Balanced DataFrame with equal number of samples for both classes
    
    Raises:
        ValueError: If the label column doesn't contain exactly two unique classes
    """
    unique_classes = df[label_column].unique()
    
    if len(unique_classes) != 2:
        raise ValueError(
            f"Expected exactly 2 classes in '{label_column}', but found {len(unique_classes)}: {unique_classes}"
        )
    
    # Auto-detect classes if not provided
    if class1 is None or class2 is None:
        class_counts = df[label_column].value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        if class1 is None:
            class1 = minority_class
        if class2 is None:
            class2 = majority_class
    
    # Ensure class1 and class2 are different
    if class1 == class2:
        raise ValueError(f"class1 and class2 must be different, but both are '{class1}'")
    
    # Get counts for both classes
    num_class1 = df[df[label_column] == class1].shape[0]
    num_class2 = df[df[label_column] == class2].shape[0]
    
    # Determine which is minority and majority
    if num_class1 < num_class2:
        minority_class, majority_class = class1, class2
        minority_count, majority_count = num_class1, num_class2
    else:
        minority_class, majority_class = class2, class1
        minority_count, majority_count = num_class2, num_class1
    
    # Downsample majority class to match minority class size
    majority_subset = df[df[label_column] == majority_class].sample(
        minority_count, random_state=random_state
    )
    
    # Combine with all minority class samples
    minority_subset = df[df[label_column] == minority_class]
    balanced_df = pd.concat([majority_subset, minority_subset])
    
    return balanced_df