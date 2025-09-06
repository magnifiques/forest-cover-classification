import pandas as pd
from src.project.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
    def validate_all_columns(self)-> bool:
        try:
            validation_status = True
            errors = []
          
            data = pd.read_csv(self.config.unzip_data_dir)
            
            # Get actual columns and their dtypes
            all_cols = dict(data.dtypes)  # {col_name: dtype}

            # Get schema columns and dtypes
            all_schema = self.config.all_schema  # {col_name: dtype}

            
            for col, dtype in all_cols.items():
                if col not in all_schema:
                    validation_status = False
                    errors.append(f"Column missing in schema: {col}")
                elif str(dtype) != str(all_schema[col]):
                    validation_status = False
                    errors.append(f"Dtype mismatch for '{col}': expected {all_schema[col]}, got {dtype}")

            # Also check if any schema columns are missing in the dataset
            for col in all_schema:
                if col not in all_cols:
                    validation_status = False
                    errors.append(f"Column missing in dataset: {col}")

            # Save validation status
            with open(self.config.STATUS_FILE, 'w') as f:
                if validation_status:
                    f.write(f"Data Validation found no errors!\n")
                    f.write(f"Validation Status: {validation_status}")
                    
                if errors:
                    f.write('Data Validation found errors! Check STATUS_FILE for details.\n')
                    f.write("Errors:\n")
                    for err in errors:
                        f.write(f"{err}\n")
                    f.write(f"Validation Status: {validation_status}")
                    
            
            return validation_status

        except Exception as e:
            raise e