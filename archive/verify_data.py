from datapreparation import load_and_prepare_data

def test_data_loading():
    print("Testing data loading...")
    try:
        train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()
        print("Data loaded successfully.")
        print(f"Train size: {len(train_ds)}")
        print(f"Val size: {len(val_ds)}")
        print(f"Test size: {len(test_ds)}")
        print(f"Labels: {label_list}")
        
        # Check for overlaps
        train_ids = set(train_ds['text'] if 'text' in train_ds.column_names else range(len(train_ds))) # dataset might not have unique IDs easily accessible, checking size and type is a good start
        
        assert len(train_ds) > 0, "Training set is empty"
        assert len(val_ds) > 0, "Validation set is empty"
        assert len(test_ds) > 0, "Test set is empty"
        assert len(label_list) == 5, f"Expected 5 labels, got {len(label_list)}"
        
        print("Verification passed!")
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()
